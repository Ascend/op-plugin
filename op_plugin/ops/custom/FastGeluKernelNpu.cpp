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
using npu_preparation = at_npu::native::OpPreparation;
using torch::autograd::AutogradContext;

namespace{
at::Tensor& fast_gelu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("FastGeluGrad")
      .Input(grad)
      .Input(self)
      .Output(grad_input)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor npu_fast_gelu(const at::Tensor& self) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  at_npu::native::OpCommand cmd;
  cmd.Name("FastGelu")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}

at::Tensor npu_fast_gelu_backward(
    const at::Tensor& grad, 
    const at::Tensor& self) {
  at::Tensor grad_input = npu_preparation::ApplyTensor(self);
  fast_gelu_backward_npu_nocheck(grad_input, grad, self);
  return grad_input;
}

class NPUFastGeluFunction : public torch::autograd::Function<NPUFastGeluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return op_plugin::npu_fast_gelu(self);
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    at::Tensor result = op_plugin::npu_fast_gelu_backward(grad_outputs[0], input);
    std::vector<at::Tensor> output = {result};
    return output;
  }
};

at::Tensor fast_gelu(const at::Tensor& self) {
    return NPUFastGeluFunction::apply(self);
}
} // namespace op_plugin
