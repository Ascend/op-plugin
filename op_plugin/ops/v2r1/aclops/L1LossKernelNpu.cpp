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

#include <torch/csrc/autograd/custom_function.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

at::Tensor& l1_loss_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    const int64_t reduction) {
  std::string reduction_str = calcu_op_util::GetReductionStr(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("LpLoss")
      .Input(self)
      .Input(target)
      .Attr("reduction", reduction_str)
      .Attr("p", (int64_t)1)
      .Output(result)
      .Run();
  return result;
}

at::Tensor npu_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::input_same_output_size(self);
  }
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  l1_loss_out_nocheck(result, self, target, reduction);
  return result;
}

at::Tensor& l1_loss_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const int64_t reduction) {
  std::string reduction_str = calcu_op_util::GetReductionStr(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("L1LossGrad")
      .Input(grad_output)
      .Input(self)
      .Input(target)
      .Attr("reduction", reduction_str)
      .Output(grad_input)
      .Run();
  return grad_input;
}

at::Tensor npu_l1_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor grad_output_broadcast = grad_output;
  at::Tensor target_broadcast = target;
  if (grad_output.sizes() != self.sizes()) {
    grad_output_broadcast = op_plugin::npu_broadcast(grad_output, self.sizes());
  }
  if (target.sizes() != self.sizes()) {
    target_broadcast = op_plugin::npu_broadcast(target, self.sizes());
  }
  at::Tensor result = npu_preparation::apply_tensor(self);
  l1_loss_backward_out_nocheck(result, grad_output_broadcast, self, target_broadcast, reduction);
  return result;
}

class NPUL1LossFunction : public torch::autograd::Function<NPUL1LossFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx,
      const at::Tensor& self,
      const at::Tensor& target,
      int64_t reduction) {
    at::AutoNonVariableTypeMode g;
    auto result = npu_l1_loss(self, target, reduction);
    ctx->save_for_backward({self, target});
    ctx->saved_data["reduction"] = reduction;
    return result;
  }

  static tensor_list backward(
      AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto reduction = ctx->saved_data["reduction"].toInt();
    auto grad_input = npu_l1_loss_backward(grad_outputs[0], saved[0], saved[1], reduction);
    tensor_list output = {grad_input, -grad_input, at::Tensor()};
    return output;
  }
};

at::Tensor l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  return NPUL1LossFunction::apply(self, target, reduction);
}
} // namespace op_plugin
