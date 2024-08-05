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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& smooth_l1_loss_backward_out(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& grad_input)
{
  DO_COMPATIBILITY(aclnnSmoothL1LossBackward,
                   acl_op::smooth_l1_loss_backward_out(grad_out, self, target, reduction, beta, grad_input));
  auto mid_shape = op_infer::broadcast_ops_npu_output_size(self.sizes(), target.sizes());
  auto output_size = op_infer::broadcast_ops_npu_output_size(mid_shape, grad_out.sizes());
  npu_preparation::check_tensor({grad_out, self, target}, grad_input, grad_input.scalar_type(), output_size);
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1LossBackward, grad_out, self, target, reduction, sigma, grad_input);
  return grad_input;
}

at::Tensor smooth_l1_loss_backward(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta)
{
  DO_COMPATIBILITY(aclnnSmoothL1LossBackward,
                   acl_op::smooth_l1_loss_backward(grad_out, self, target, reduction, beta));
  auto mid_shape = op_infer::broadcast_ops_npu_output_size(self.sizes(), target.sizes());
  auto output_size = op_infer::broadcast_ops_npu_output_size(mid_shape, grad_out.sizes());
  at::Tensor grad_input = npu_preparation::apply_tensor_without_format(self, output_size);
  float sigma = static_cast<float>(beta);
  EXEC_NPU_CMD(aclnnSmoothL1LossBackward, grad_out, self, target, reduction, sigma, grad_input);
  return grad_input;
}

}
