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
using tensor_list = std::vector<at::Tensor>;
using torch::autograd::AutogradContext;

namespace {
at::Tensor mish_npu(const at::Tensor& self) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  at_npu::native::OpCommand cmd;
  cmd.Name("Mish")
      .Input(self)
      .Output(result)
      .Run();
  return result;
}
} // namespace

class NPUMishFunction : public torch::autograd::Function<NPUMishFunction> {
public:
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& self) {
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self});
    return mish_npu(self);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];

    at::Tensor result = op_plugin::npu_mish_backward(grad_outputs[0], input);
    tensor_list output = {result};
    return output;
  }
};

at::Tensor npu_mish_backward(const at::Tensor& grad, const at::Tensor& input) {
  at::Tensor result = npu_preparation::ApplyTensor(input);
  at_npu::native::OpCommand cmd;
  cmd.Name("MishGrad")
      .Input(grad)
      .Input(input)
      .Output(result)
      .Run();
  return result;
}

at::Tensor npu_mish(const at::Tensor& self) {
  return NPUMishFunction::apply(self);
}
} // namespace op_plugin
