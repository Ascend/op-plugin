// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

at::Tensor npu_transpose_batchmatmul(
    const at::Tensor &input, const at::Tensor &weight,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &scale,
    c10::OptionalIntArrayRef perm_x1, c10::OptionalIntArrayRef perm_x2,
    c10::OptionalIntArrayRef perm_y, c10::optional<int64_t> batch_split_factor
    )
{
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &scale_real = scale.value_or(at::Tensor());
    int32_t batch_split_factor_value = static_cast<int32_t>(batch_split_factor.value_or(1));
    auto output_size = op_infer::array_to_small_vector(input.sizes());

    auto perm_x1_real = at::IntArrayRef{0, 1, 2};
    auto perm_x2_real = at::IntArrayRef{0, 1, 2};
    auto perm_y_real = at::IntArrayRef{0, 1, 2};

    if (perm_x1.has_value()) {
        perm_x1_real = perm_x1.value();
    }
    if (perm_x2.has_value()) {
        perm_x2_real = perm_x2.value();
    }
    if (perm_y.has_value()) {
        perm_y_real = perm_y.value();
    }

    auto input_dim_num = input.dim();
    auto weight_dim_num = weight.dim();
    constexpr int EXPECTED_DIM = 3;

    TORCH_CHECK(input_dim_num == EXPECTED_DIM && weight_dim_num == EXPECTED_DIM, "input dim is ", EXPECTED_DIM,
                "weight dim is ", EXPECTED_DIM, OPS_ERROR(ErrCode::PARAM));

    auto m_dim = input.size(perm_x1_real[1]);
    auto batch_dim = input.size(perm_x1_real[0]);
    auto n_dim = weight.size(perm_x2_real[2]);

    constexpr int DIM_0 = 0;
    constexpr int DIM_1 = 1;
    constexpr int DIM_2 = 2;
    if (batch_split_factor_value > 1) {
        output_size[DIM_0] = batch_split_factor_value;
        output_size[DIM_1] = m_dim;
        output_size[DIM_2] = batch_dim * n_dim / batch_split_factor_value;
    } else {
        output_size[DIM_0] = m_dim;
        output_size[DIM_1] = batch_dim;
        output_size[DIM_2] = n_dim;
    }

    at::ScalarType dst_type = at::ScalarType::Half;
    if (scale_real.defined()) {
        dst_type = at::ScalarType::Int;
    }
    c10::TensorOptions options = input.options().dtype(dst_type);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    int8_t cubeMathType = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());

    EXEC_NPU_CMD(aclnnTransposeBatchMatmul, input, weight, bias_real, scale_real, perm_x1_real,
                 perm_x2_real, perm_y_real, cubeMathType, batch_split_factor_value, result);
    return result;
}
}
