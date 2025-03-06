// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &avg_pool2d_backward_out_npu_nocheck_api(const at::Tensor &grad_output, const at::Tensor &self,
                                                    at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                    at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                                                    c10::optional<int64_t> divisor_override, at::Tensor &grad_input) {
  int64_t new_divisor_override = divisor_override.has_value() ? divisor_override.value() : 0;
  int8_t cube_math_type = npu_preparation::get_cube_math_type(false);
  EXEC_NPU_CMD(aclnnAvgPool2dBackward, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
               new_divisor_override, cube_math_type, grad_input);
  return grad_input;
}

at::Tensor &avg_pool2d_backward_out(const at::Tensor &grad_output, const at::Tensor &self,
                                    at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                    at::IntArrayRef padding, bool ceil_mode,
                                    bool count_include_pad,
                                    c10::optional<int64_t> divisor_override,
                                    at::Tensor &grad_input)
{
    DO_COMPATIBILITY(aclnnAvgPool2dBackward, acl_op::avg_pool2d_backward_out(grad_output, self, kernel_size, stride,
                                                                             padding, ceil_mode, count_include_pad, divisor_override, grad_input));
    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero", OPS_ERROR(ErrCode::VALUE));

    auto input_size = op_infer::avg_pool2d_backward_npu_output_size(self);
    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    avg_pool2d_backward_out_npu_nocheck_api(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                            divisor_override, grad_input);

    return grad_input;
}

at::Tensor avg_pool2d_backward(const at::Tensor &grad_output, const at::Tensor &self,
                               at::IntArrayRef kernel_size, at::IntArrayRef stride,
                               at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override)
{
    DO_COMPATIBILITY(aclnnAvgPool2dBackward, acl_op::avg_pool2d_backward(grad_output, self, kernel_size, stride,
                                                                         padding, ceil_mode, count_include_pad, divisor_override));
    TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(self.dim() == 3 || self.dim() == 4, "tensor self's dimension must be 3 or 4", OPS_ERROR(ErrCode::VALUE));

    auto input_size = op_infer::avg_pool2d_backward_npu_output_size(self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    avg_pool2d_backward_out_npu_nocheck_api(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad,
                                            divisor_override, grad_input);

    return grad_input;
}

}
