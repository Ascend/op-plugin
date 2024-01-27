// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
constexpr size_t X_MIN_DIM = 2;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t X_MAX_DIM = 6;
using npu_preparation = at_npu::native::OpPreparation;

uint64_t infer_out_batch_shape(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
{
    uint64_t batch_val = 1;
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto out_dim_num = std::max(x1_dim_num, x2_dim_num);
    auto &shape_long = x1_dim_num > x2_dim_num ? x1 : x2;
    auto &shape_short = x1_dim_num > x2_dim_num ? x2 : x1;
    size_t vaild_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (size_t i = 0; i < out_dim_num - LAST_SECOND_DIM_INDEX; i++) {
        auto short_dim = i < vaild_offset ? 1 : shape_short.size(i - vaild_offset);
        auto long_dim = shape_long.size(i);
        TORCH_CHECK(!(short_dim > 1 && long_dim > 1 && short_dim != long_dim),
                    "the x1 shape and x2 shape not supported for broadcast, the short_dim is ",
                    short_dim, " and  the long_dim is ", long_dim);
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_val = batch_val * cur_batch_value;
        batch_record.push_back(cur_batch_value);
    }
    return batch_val;
}

void bias_shape_check(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &bias, uint64_t batch_val)
{
    auto x2_dim_num = x2.dim();
    auto x2_n_dim = x2.size(x2_dim_num - 1);
    auto bias_dim_num = bias.dim();
    TORCH_CHECK(bias_dim_num == 1 || bias_dim_num == 3, "bias dim num should be 1 or 3, but actual bias_dim_num is ",
                bias_dim_num);
    auto bias_first_dim = bias.size(0);
    if (bias_dim_num == 1) {
        TORCH_CHECK(bias_first_dim == x2_n_dim,
                    "bias_first_dim should be equal to x2 n dim . bias_first_dim is ", bias_first_dim,
                    " and x2_n_dim is ", x2_n_dim);
        return;
    }
    auto bias_second_dim = bias.size(1);
    auto bias_third_dim = bias.size(2);
    TORCH_CHECK(bias_first_dim == batch_val,
                "infered batch value should be equal to bias batch dim value. batch infered value is ", batch_val,
                " and bias batch dim value is ", bias_first_dim);
    TORCH_CHECK(bias_second_dim == 1, "second dim of bias should be 1, but bias_second_dim is ", bias_second_dim);
    TORCH_CHECK(bias_third_dim == x2_n_dim, "third dim should be equal to n, but bias_third_dim is ",
                bias_third_dim, " and n dim is ", x2_n_dim);
}

at::Tensor npu_quant_matmul(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& scale,
                            const c10::optional<at::Tensor>& offset, const c10::optional<at::Tensor>& bias,
                            c10::optional<c10::string_view> output_dtype)
{
    auto x1_dim_num = x1.dim();
    TORCH_CHECK(x1_dim_num >= X_MIN_DIM && x1_dim_num <= X_MAX_DIM, "x1 shape dim num should be within 2~6, but it is ",
                x1_dim_num);
    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num >= X_MIN_DIM && x2_dim_num <= X_MAX_DIM, "x2 shape dim num should be within 2~6, but it is ",
                x2_dim_num);
    TORCH_CHECK(x1_dim_num == x2_dim_num, "x1_dim_num should be equal to x2_dim_num, but x1_dim_num is ", x1_dim_num,
                " and x2_dim_num is ", x2_dim_num);
    auto x1_k_dim = x1.size(x1_dim_num - 1);
    auto x2_n_dim = x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - 2);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ",
                x1_k_dim, ", x2_k_dim is ", x2_k_dim);

    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape(x1, x2, batch_record);
    const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x1 : x2;
    auto output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
    output_size[long_tensor.dim() - 1] = x2.size(x2_dim_num - 1);
    for (size_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = batch_record[i];
    }
    c10::TensorOptions options;
    if (!output_dtype.has_value() ||  *output_dtype == "int8") {
        options = x1.options().dtype(at::kChar);
    } else if (*output_dtype == "float16") {
        options = x1.options().dtype(at::kHalf);
    } else if (*output_dtype == "bfloat16") {
        options = x1.options().dtype(at::kBFloat16);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    bool transpose1 = false;
    bool transpose2 = false;

    auto scale_dim_num = scale.dim();
    TORCH_CHECK(scale_dim_num == 1, "The scale dim num should be 1. but scale_dim_num is ", scale_dim_num);
    auto scale_first_dim = scale.size(0);
    TORCH_CHECK(scale_first_dim == 1 || scale_first_dim == x2_n_dim,
                "The scale 1st dim should be 1 or n, but scale_first_dim is ", scale_first_dim);

    if (offset.has_value()) {
        auto offset_dim_num = offset_real.dim();
        TORCH_CHECK(offset_dim_num == 1, "The offset dim num should be 1. but offset_dim_num is ", offset_dim_num);
        auto offset_first_dim_value = offset_real.size(0);
        TORCH_CHECK(offset_first_dim_value == 1 || offset_first_dim_value == x2_n_dim,
                    "The offset 1st dim should be 1 or n, but offset_first_dim is ", offset_first_dim_value);
    }

    if (bias.has_value()) {
        bias_shape_check(x1, x2, bias_real, batch_val);
    }

    if (scale.dtype() == at::kFloat) {
        const at::Tensor quant_param = op_api::npu_trans_quant_param(scale, offset);
        EXEC_NPU_CMD(aclnnQuantMatmulV3, x1, x2, quant_param, offset_real, bias_real, transpose1, transpose2, result);
    } else {
        EXEC_NPU_CMD(aclnnQuantMatmulV3, x1, x2, scale, offset_real, bias_real, transpose1, transpose2, result);
    }
    return result;
}
}  // namespace op_api