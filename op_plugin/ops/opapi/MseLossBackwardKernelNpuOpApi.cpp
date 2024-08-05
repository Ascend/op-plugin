// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

namespace op_api {

at::Tensor& mse_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnMseLossBackward,
                   acl_op::mse_loss_backward_out(grad_output, self, target, reduction, grad_input));
  auto output_size_pre = op_infer::broadcast_ops_npu_output_size(grad_output.sizes(), self.sizes());
  auto output_size = op_infer::broadcast_ops_npu_output_size(output_size_pre, target.sizes());
  at_npu::native::OpPreparation::check_tensor(
      {grad_output, self, target}, grad_input, grad_input.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnMseLossBackward, grad_output, self, target, reduction, grad_input);
  return grad_input;
}

at::Tensor mse_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  DO_COMPATIBILITY(aclnnMseLossBackward,
                   acl_op::mse_loss_backward(grad_output, self, target, reduction));
  auto output_size_pre = op_infer::broadcast_ops_npu_output_size(grad_output.sizes(), self.sizes());
  auto output_size = op_infer::broadcast_ops_npu_output_size(output_size_pre, target.sizes());
  at::Tensor grad_input = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnMseLossBackward, grad_output, self, target, reduction, grad_input);
  return grad_input;
}

}
