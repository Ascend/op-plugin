// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
constexpr int MINIMUM_SHAPE_SIZE = 2;
const int64_t INT4_NUMS_IN_INT32 = 8;
at::Tensor npu_weight_quant_batchmatmul(const at::Tensor &x, const at::Tensor &weight,
                                        const at::Tensor &antiquant_scale,
                                        const c10::optional<at::Tensor> &antiquant_offset,
                                        const c10::optional<at::Tensor> &quant_scale,
                                        const c10::optional<at::Tensor> &quant_offset,
                                        const c10::optional<at::Tensor> &bias,
                                        int64_t antiquant_group_size,
                                        int64_t inner_precise)
{
    bool trans_weight = op_plugin::utils::is_transpose_last_two_dims(weight);
    auto x_dim_num = x.dim();
    auto weight_dim_num = weight.dim();
    TORCH_CHECK(x_dim_num >= MINIMUM_SHAPE_SIZE, "x shape do not support dim num less than 2, but it is ", x_dim_num,
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight_dim_num >= MINIMUM_SHAPE_SIZE, "weight shape do not support dim num less than 2, but it is ",
                weight_dim_num, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!(std::min(x_dim_num, weight_dim_num) > MINIMUM_SHAPE_SIZE && x_dim_num != weight_dim_num),
                "x dim is not the same as weight dim", OPS_ERROR(ErrCode::PARAM));
    auto x_k_dim = x.size(x_dim_num - 1);
    auto weight_k_dim = (weight.dtype() == at::kInt && trans_weight) ?
                        weight.size(weight_dim_num - MINIMUM_SHAPE_SIZE) * INT4_NUMS_IN_INT32 :
                        weight.size(weight_dim_num - MINIMUM_SHAPE_SIZE);
    TORCH_CHECK(x_k_dim == weight_k_dim, "The k of x and weight should be equal. but x_k_dim is ", x_k_dim,
                ", weight_k_dim is ", weight_k_dim, OPS_ERROR(ErrCode::PARAM));
    auto out_dim_num = std::max(x_dim_num, weight_dim_num);
    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[out_dim_num - MINIMUM_SHAPE_SIZE] = x.size(x_dim_num - MINIMUM_SHAPE_SIZE);
    auto weight_size_base = weight.size(weight_dim_num - MINIMUM_SHAPE_SIZE + 1);
    output_size[out_dim_num - MINIMUM_SHAPE_SIZE + 1] = (weight.dtype() == at::kInt && !trans_weight) ?
                                                        weight_size_base * INT4_NUMS_IN_INT32 :
                                                        weight_size_base;
    if (x_dim_num == weight_dim_num) {
        for (auto i = 0; i < out_dim_num - MINIMUM_SHAPE_SIZE; i++) {
            TORCH_CHECK(x.size(i) == weight.size(i), "batch of x is diff from batch of weight",
                        OPS_ERROR(ErrCode::PARAM));
            output_size[i] = x.size(i);
        }
    } else {
        auto longer_tensor = x_dim_num > weight_dim_num ? x : weight;
        for (auto i = 0; i < out_dim_num - MINIMUM_SHAPE_SIZE; i++) {
            output_size[i] = longer_tensor.size(i);
        }
    }
    const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
    const at::Tensor &quant_scale_real = quant_scale.value_or(at::Tensor());
    const at::Tensor &quant_offset_real = quant_offset.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int antiquant_group_size_real = static_cast<int>(antiquant_group_size);
    bool is_group_size_vaild = antiquant_group_size_real == 0 || (antiquant_group_size_real >= 32 &&
                antiquant_group_size_real <= weight_k_dim - 1 && antiquant_group_size_real % 32 == 0);
    TORCH_CHECK(is_group_size_vaild,
                "antiquant_group_size can be either 0 or a multiple of 32 within the range 32 to weight_k_dim - 1.",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((quant_scale.has_value() || !quant_offset.has_value()),
                "Quantization parameters are incorrectly set, quant_offset cannot exist in isolation from quant_scale",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((inner_precise == 0 ||  inner_precise == 1),
                "inner_precise only support 0 or 1. but is:", inner_precise,
                OPS_ERROR(ErrCode::PARAM));

    c10::TensorOptions options =
        quant_scale.has_value() ? x.options().dtype(at::kChar) : x.options().dtype(x.scalar_type());
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    if (quant_scale.has_value() && quant_scale_real.dtype() == at::kFloat) {
        auto quant_scale_output_size = op_infer::array_to_small_vector(quant_scale_real.sizes());
        c10::TensorOptions quant_scale_options = quant_scale_real.options().dtype(at::kLong);
        at::Tensor quant_scale_result = npu_preparation::apply_tensor_without_format(quant_scale_output_size,
                                                                                     quant_scale_options);
        EXEC_NPU_CMD(aclnnTransQuantParamV2, quant_scale_real, quant_offset_real, quant_scale_result);
        EXEC_NPU_CMD(aclnnWeightQuantBatchMatmulV2, x, weight, antiquant_scale, antiquant_offset_real,
                     quant_scale_result, quant_offset_real, bias_real, antiquant_group_size_real, result);
    } else if (inner_precise == 1) { // 1: high performance mode
        EXEC_NPU_CMD(aclnnWeightQuantBatchMatmulV3, x, weight, antiquant_scale, antiquant_offset_real, quant_scale_real,
                     quant_offset_real, bias_real, antiquant_group_size_real, inner_precise, result);
    } else {
        EXEC_NPU_CMD(aclnnWeightQuantBatchMatmulV2, x, weight, antiquant_scale, antiquant_offset_real, quant_scale_real,
                     quant_offset_real, bias_real, antiquant_group_size_real, result);
    }

    return result;
}
}  // namespace op_api
