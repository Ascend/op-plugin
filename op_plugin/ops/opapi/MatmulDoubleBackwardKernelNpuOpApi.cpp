// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
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
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "op_plugin/utils/custom_functions/opapi/inner_compute_op_api.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const int8_t ALLOW_FP32_DOWN_PRECISION = 1;
const int8_t KEEP_DTYPE = 0;

static inline void matmul_implement_npu(at::Tensor &out,
                                        const at::Tensor &self,
                                        const at::Tensor &mat2)
{
    // allow dicrease precision
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMatmul, self, mat2, out, cube_math_type);
    FLOP_COUNT(FlopCounter::mm_flop, self, mat2);
    return;
}

at::Tensor matmul_grad_double_backward(const at::Tensor &self,
                                       const at::Tensor &other,
                                       const at::Tensor &grad_self,
                                       const at::Tensor &grad_other,
                                       const at::Tensor &grad_out)
{
    /* grad_grad = grad_self * other + self * grad_other */
    at::Tensor grad_self_to_grad = at::zeros_like(grad_out);
    at::Tensor grad_other_to_grad = at::zeros_like(grad_out);
    if (grad_self.defined()) {
        matmul_implement_npu(grad_self_to_grad, grad_self, other);
    }
    if (grad_other.defined()) {
        matmul_implement_npu(grad_other_to_grad, self, grad_other);
    }
    return grad_self_to_grad + grad_other_to_grad;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> matmul_double_backward(const c10::optional<at::Tensor> &grad_self_opt,
                                                                      const c10::optional<at::Tensor> &grad_other_opt,
                                                                      const at::Tensor &grad_out,
                                                                      const at::Tensor &self,
                                                                      const at::Tensor &other,
                                                                      std::array<bool, 3> grad_input_mask)
{
    if (!grad_out.defined()) {
        return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
    }

    at::Tensor grad_self = grad_self_opt.value_or(at::Tensor());
    at::Tensor grad_other = grad_other_opt.value_or(at::Tensor());

    if (grad_self.defined() && grad_self.dim() == self.dim() + 1) {
        grad_self = grad_self[0];
    }
    if (grad_other.defined() && grad_other.dim() == other.dim() + 1) {
        grad_other = grad_other[0];
    }

    at::Tensor grad_grad;
    at::Tensor self_grad;
    at::Tensor other_grad;

    if (grad_input_mask[0] && (grad_self.defined() || grad_other.defined())) {
        grad_grad = matmul_grad_double_backward(self, other, grad_self, grad_other, grad_out);
    }
    if (grad_input_mask[1] && grad_other.defined()) {
        /* self_grad = grad_out * grad_other^T
        Because matmul_mat1_backward(mat1, mat2, grad) calculates mat1_grad = grad * mat2^T, we have: */
        self_grad = op_api::matmul_mat1_backward(self, grad_other, grad_out);
    }
    if (grad_input_mask[2] && grad_self.defined()) {
        /* other_grad = grad_self^T * grad_out
        Because matmul_mat2_backward(mat1, mat2, grad) calculates mat2_grad = mat1^T * grad, we have: */
        other_grad = op_api::matmul_mat2_backward(grad_self, other, grad_out);
    }

    // strip added dim: (5,1)->(5)
    if (other.dim() == 1 && other_grad.size(-1) == 1 && other_grad.dim() != 1) {
        other_grad = other_grad.squeeze(-1);
    }

    return std::make_tuple(grad_grad, self_grad, other_grad);
}

} // namespace op_api

