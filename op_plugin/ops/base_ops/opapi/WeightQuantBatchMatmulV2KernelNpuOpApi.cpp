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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_weight_quant_batchmatmul(const at::Tensor &x, const at::Tensor &weight,
                                        const at::Tensor &antiquant_scale,
                                        const c10::optional<at::Tensor> &antiquant_offset,
                                        const c10::optional<at::Tensor> &quant_scale,
                                        const c10::optional<at::Tensor> &quant_offset,
                                        const c10::optional<at::Tensor> &bias,
                                        int64_t antiquant_group_size)
{
    auto x_dim_num = x.dim();
    auto weight_dim_num = weight.dim();
    TORCH_CHECK(x_dim_num == 2, "x shape dims should be 2, but it is ", x_dim_num);
    TORCH_CHECK(weight_dim_num == 2, "weight shape dims should be 2, but it is ", weight_dim_num);

    auto x_k_dim = x.size(1);
    auto weight_k_dim = weight.size(0);
    TORCH_CHECK(x_k_dim == weight_k_dim, "The k of x and weight should be equal. but x_k_dim is ", x_k_dim,
                ", weight_k_dim is ", weight_k_dim);

    const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
    const at::Tensor &quant_scale_real = quant_scale.value_or(at::Tensor());
    const at::Tensor &quant_offset_real = quant_offset.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int antiquant_group_size_real = static_cast<int>(antiquant_group_size);
    bool is_group_size_vaild = antiquant_group_size_real == 0 || (antiquant_group_size_real >= 32 &&
                antiquant_group_size_real <= weight_k_dim - 1 && antiquant_group_size_real != 0 &&
                antiquant_group_size_real % 32 == 0);
    TORCH_CHECK(is_group_size_vaild,
                "antiquant_group_size can be either 0 or a multiple of 32 within the range 32 to weight_k_dim - 1.");
    TORCH_CHECK((quant_scale.has_value() || !quant_offset.has_value()),
                "Quantization parameters are incorrectly set, quant_offset cannot exist in isolation from quant_scale");

    c10::TensorOptions options =
        quant_scale.has_value() ? x.options().dtype(at::kChar) : x.options().dtype(x.scalar_type());

    auto output_size = op_infer::array_to_small_vector(x.sizes());
    output_size[0] = x.size(0);
    output_size[1] = weight.size(1);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    EXEC_NPU_CMD(aclnnWeightQuantBatchMatmulV2, x, weight, antiquant_scale, antiquant_offset_real, quant_scale_real,
                 quant_offset_real, bias_real, antiquant_group_size_real, result);

    return result;
}
}  // namespace op_api
