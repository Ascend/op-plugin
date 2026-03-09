// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

c10::SmallVector<int64_t, SIZE> cal_output_size(const at::Tensor &input, const at::Tensor &weight,
                                                const at::Tensor &scale_real, at::IntArrayRef perm_x1_real,
                                                at::IntArrayRef perm_x2_real, int64_t batch_split_factor_value)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    
    auto m_dim = input.size(perm_x1_real[1]);
    auto batch_dim = input.size(perm_x1_real[0]);
    auto n_dim = weight.size(perm_x2_real[2]);

    output_size = {m_dim, batch_dim, n_dim};
    if (scale_real.defined()) {
        output_size = {m_dim, 1, batch_dim * n_dim};
    }

    if (batch_split_factor_value > 1) {
        output_size = {batch_split_factor_value, m_dim, batch_dim * n_dim / batch_split_factor_value};
    }
    return output_size;
}

at::Tensor npu_transpose_batchmatmul(const at::Tensor &input, const at::Tensor &weight,
                                     const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &scale,
                                     at::OptionalIntArrayRef perm_x1, at::OptionalIntArrayRef perm_x2,
                                     at::OptionalIntArrayRef perm_y, c10::optional<int64_t> batch_split_factor)
{
    // Use the correct function call with all required parameters
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &scale_real = scale.value_or(at::Tensor());
    const auto &perm_x1_real = perm_x1.value_or(at::IntArrayRef({0, 1, 2}));
    const auto &perm_x2_real = perm_x2.value_or(at::IntArrayRef({0, 1, 2}));
    const auto &perm_y_real = perm_y.value_or(at::IntArrayRef({1, 0, 2}));
    int64_t batch_split_factor_value = batch_split_factor.value_or(1);
    auto output_size = cal_output_size(input, weight, scale_real, perm_x1_real, perm_x2_real, batch_split_factor_value);
    
    // Construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, input.options());
    int cubeMathType = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    // Check format and execute the appropriate NPU command
    bool is_nd_nz_format = op_plugin::utils::is_nz_format(weight) && !op_plugin::utils::is_nz_format(input);
    if (is_nd_nz_format) {
        EXEC_NPU_CMD(aclnnTransposeBatchMatMulWeightNz, input, weight, bias_real, scale_real, perm_x1_real,
                     perm_x2_real, perm_y_real, cubeMathType, batch_split_factor_value, result);
    } else {
        EXEC_NPU_CMD(aclnnTransposeBatchMatMul, input, weight, bias_real, scale_real, perm_x1_real, perm_x2_real,
                     perm_y_real, cubeMathType, batch_split_factor_value, result);
    }

    return result;
}
} // namespace op_api
