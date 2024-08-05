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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor&, at::Tensor&> log_sigmoid_forward_out(const at::Tensor &self,
                                                             at::Tensor &out,
                                                             at::Tensor &buffer)
{
  DO_COMPATIBILITY(aclnnLogSigmoid, acl_op::log_sigmoid_forward_out(self, out, buffer));
  at_npu::native::OpPreparation::check_tensor({self}, out, self);
  EXEC_NPU_CMD(aclnnLogSigmoidForward, self, out, buffer);
  return std::tie(out, buffer);
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(const at::Tensor &self)
{
  DO_COMPATIBILITY(aclnnLogSigmoid, acl_op::log_sigmoid_forward(self));
  at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(self);
  at::Tensor buffer = at_npu::native::OpPreparation::apply_tensor_with_sizes({0}, self.options());
  EXEC_NPU_CMD(aclnnLogSigmoidForward, self, out, buffer);
  return std::tuple<at::Tensor, at::Tensor>(out, buffer);
}

at::Tensor& log_sigmoid_out(const at::Tensor &self, at::Tensor &out)
{
  DO_COMPATIBILITY(aclnnLogSigmoid, acl_op::log_sigmoid_out(self, out));
  at_npu::native::OpPreparation::check_tensor({self}, out, self);
  at::Tensor buffer = at_npu::native::OpPreparation::apply_tensor_with_sizes({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(out, buffer, self));
}

at::Tensor log_sigmoid(const at::Tensor &self)
{
  DO_COMPATIBILITY(aclnnLogSigmoid, acl_op::log_sigmoid(self));
  return std::get<0>(at::log_sigmoid_forward(self));
}

}
