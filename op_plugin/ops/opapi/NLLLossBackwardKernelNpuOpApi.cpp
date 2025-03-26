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

at::Tensor& nll_loss_backward_out(const at::Tensor& grad_output, const at::Tensor& self,
                                  const at::Tensor& target,
                                  const c10::optional<at::Tensor>& weight,
                                  int64_t reduction, int64_t ignore_index,
                                  const at::Tensor& total_weight, at::Tensor& grad_input)
{
    DO_COMPATIBILITY(aclnnNLLLossBackward,
                     acl_op::nll_loss_backward_out(grad_output, self, target, weight, reduction,
                                                   ignore_index, total_weight, grad_input));
    at::Tensor weight_tensor = c10::value_or_else(weight, [] { return at::Tensor(); });
    if (!weight_tensor.defined()) {
        weight_tensor = at::ones(self.size(-1), self.options());
    }

    EXEC_NPU_CMD(aclnnNLLLossBackward, grad_output, self, target, weight_tensor, reduction, ignore_index, total_weight,
                 grad_input);
    return grad_input;
}

at::Tensor nll_loss_backward(const at::Tensor& grad_output, const at::Tensor& self,
                             const at::Tensor& target, const c10::optional<at::Tensor>& weight,
                             int64_t reduction, int64_t ignore_index,
                             const at::Tensor& total_weight)
{
    DO_COMPATIBILITY(aclnnNLLLossBackward, acl_op::nll_loss_backward(grad_output, self, target, weight,
                                                                     reduction, ignore_index, total_weight));
    // construct the output tensor of the NPU
    at::Tensor grad_input = at_npu::native::OpPreparation::apply_tensor_without_format(self);

    // calculate the output result of the NPU
    nll_loss_backward_out(grad_output, self, target, weight, reduction, ignore_index,
                          total_weight, grad_input);

    return grad_input;
}

}
