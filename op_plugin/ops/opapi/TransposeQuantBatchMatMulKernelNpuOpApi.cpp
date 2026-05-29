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
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

c10::SmallVector<int64_t, SIZE> npu_transpose_quant_batchmatmul_output_size(const at::Tensor &x1, const at::Tensor &x2,
    int64_t dtype, const at::Tensor &x1_scale_real, const at::Tensor &x2_scale_real, int32_t group_size_value,
    at::IntArrayRef perm_x1_real, at::IntArrayRef perm_x2_real, at::IntArrayRef perm_y_real,
    int64_t batch_split_factor_value)
{
    c10::SmallVector<int64_t, SIZE> output_size;
    auto m_dim = x1.size(perm_x1_real[1]);
    auto batch_dim = x1.size(perm_x1_real[0]);
    auto n_dim = x2.size(perm_x2_real[2]);

    output_size = {m_dim, batch_dim, n_dim};

    if (batch_split_factor_value > 1) {
        output_size = {batch_split_factor_value, m_dim, batch_dim * n_dim / batch_split_factor_value};
    }
    return output_size;
}

at::Tensor npu_transpose_quant_batchmatmul(const at::Tensor &x1, const at::Tensor &x2, int64_t dtype,
                                           const c10::optional<at::Tensor> &bias,
                                           const c10::optional<at::Tensor> &x1_scale,
                                           const c10::optional<at::Tensor> &x2_scale,
                                           at::OptionalIntArrayRef group_sizes,
                                           at::OptionalIntArrayRef perm_x1,
                                           at::OptionalIntArrayRef perm_x2,
                                           at::OptionalIntArrayRef perm_y,
                                           c10::optional<int64_t> batch_split_factor,
                                           c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype)
{
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &x1_scale_real = x1_scale.value_or(at::Tensor());
    const at::Tensor &x2_scale_real = x2_scale.value_or(at::Tensor());
    const int64_t b_idx = 0;
    const int64_t m_idx = 1;
    const int64_t ka_idx = 2;
    const int64_t kb_idx = 1;
    const int64_t n_idx = 2;
    const auto default_perm_x1 = std::vector<int64_t>{m_idx, b_idx, ka_idx};
    const auto default_perm_x2 = std::vector<int64_t>{b_idx, kb_idx, n_idx};
    const auto default_perm_y = std::vector<int64_t>{m_idx, b_idx, n_idx};
    const auto default_group_sizes = std::vector<int64_t>{0, 0, 0};
    
    const auto perm_x1_real = perm_x1.value_or(at::IntArrayRef(default_perm_x1));
    const auto perm_x2_real = perm_x2.value_or(at::IntArrayRef(default_perm_x2));
    const auto perm_y_real = perm_y.value_or(at::IntArrayRef(default_perm_y));
    int64_t group_size_value =
        op_plugin::utils::check_and_get_group_size(group_sizes.value_or(at::IntArrayRef(default_group_sizes)));
    int64_t batch_split_factor_value = batch_split_factor.value_or(1);
    
    auto output_size = npu_transpose_quant_batchmatmul_output_size(
        x1, x2, dtype, x1_scale_real, x2_scale_real, group_size_value,
        perm_x1_real, perm_x2_real, perm_y_real, batch_split_factor_value);
    
    aclDataType dtype_value = c10_npu::GetAclDataType(dtype);
    at::ScalarType scalar_type = npu_preparation::convert_to_scalar_type(dtype_value);
    
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(scalar_type));
    
    
    bool is_nd_nz_format = op_plugin::utils::is_nz_format(x2) && !op_plugin::utils::is_nz_format(x1);
    TensorWrapper x1_wrapper = make_wrapper(x1, x1_dtype);
    TensorWrapper x2_wrapper = make_wrapper(x2, x2_dtype);
    TensorWrapper result_wrapper = make_wrapper(result, dtype);
    if (is_nd_nz_format) {
        if (check_aclnn_kernel_available("aclnnTransposeQuantBatchMatMulWeightNz")) {
            EXEC_NPU_CMD(aclnnTransposeQuantBatchMatMulWeightNz, x1_wrapper, x2_wrapper, bias_real, x1_scale_real,
                         x2_scale_real, dtype_value, group_size_value, perm_x1_real, perm_x2_real, perm_y_real,
                         batch_split_factor_value, result_wrapper);
        } else {
            TORCH_CHECK(false, "In the current CANN version, aclnnTransposeQuantBatchMatMulWeightNz does not support x2 as WeightNz input. Please upgrade the CANN package to version 9.1 or higher, or set the x2 to ND mode.", OPS_ERROR(ErrCode::PARAM));
        }
    } else {
        EXEC_NPU_CMD(aclnnTransposeQuantBatchMatMul, x1_wrapper, x2_wrapper, bias_real, x1_scale_real, x2_scale_real,
                     dtype_value, group_size_value, perm_x1_real, perm_x2_real, perm_y_real,
                     batch_split_factor_value, result_wrapper);
    }
    
    return result;
}
} // namespace op_api