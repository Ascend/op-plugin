// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

at::Tensor& elu_backward_out(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                             const at::Scalar& input_scale, bool is_result, const at::Tensor& self_or_result,
                             at::Tensor & grad_input) {
  DO_COMPATIBILITY(aclnnEluBackward, acl_op::elu_backward_out(grad_output, alpha, scale, input_scale, is_result,
                                                              self_or_result, grad_input));
  npu_preparation::check_tensor({grad_output}, grad_input, grad_output.sizes());
  EXEC_NPU_CMD(aclnnEluBackward, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
  return grad_input;
}

at::Tensor elu_backward(const at::Tensor& grad_output, const at::Scalar& alpha, const at::Scalar& scale,
                        const at::Scalar& input_scale, bool is_result, const at::Tensor &self_or_result) {
  DO_COMPATIBILITY(aclnnEluBackward, acl_op::elu_backward(grad_output, alpha, scale, input_scale, is_result,
                                                          self_or_result));
  at::Tensor result = npu_preparation::apply_tensor_without_format(grad_output);
  EXEC_NPU_CMD(aclnnEluBackward, grad_output, alpha, scale, input_scale, is_result, self_or_result, result);
  return result;
}
} // namespace op_api
