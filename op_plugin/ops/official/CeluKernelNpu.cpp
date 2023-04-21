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
using npu_utils = at_npu::native::NpuUtils;
using torch::autograd::AutogradContext;
using torch::autograd::Function;

namespace{
at::Tensor& celu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar& alpha) {
  at_npu::native::OpCommand cmd;
  cmd.Name("CeluV2")
      .Input(self)
      .Output(result)
      .Attr("alpha", alpha)
      .Run();
  return result;
}

at::Tensor celu_npu_impl(const at::Tensor& self, at::Scalar& alpha) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  celu_out_npu_nocheck(result, self, alpha);
  return result;
}

at::Tensor& celu_backward_out_npu(at::Tensor& grad_input, const at::Tensor& grad_output,
    at::Scalar& alpha, const at::Tensor& output) {
  at_npu::native::OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(output)
      .Output(grad_input)
      .Attr("alpha", alpha)
      .Run();
  return grad_input;
}

at::Tensor celu_backward_npu_impl(const at::Tensor& grad_output, at::Scalar& alpha, const at::Tensor& output) {
  at::Tensor result = npu_preparation::ApplyTensor(grad_output);
  celu_backward_out_npu(result, grad_output, alpha, output);
  return result;
}
} // namespace

class NPUCeluFunction : public torch::autograd::Function<NPUCeluFunction> {
public:
  static at::Tensor forward(AutogradContext* ctx, const at::Tensor& self, at::Scalar alpha) {
    ctx->saved_data["alpha"] = alpha;
    at::AutoNonVariableTypeMode g;
    at::Tensor result = celu_npu_impl(self, alpha);
    ctx->save_for_backward({result});
    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx, std::vector<at::Tensor> grad_outputs) {
    auto alpha = ctx->saved_data["alpha"].toScalar();
    auto saved = ctx->get_saved_variables();
    auto result = saved[0];
    auto grad_input = celu_backward_npu_impl(
        grad_outputs[0],
        alpha,
        result);
    std::vector<at::Tensor> output = {grad_input, at::Tensor()};
    return output;
  }
};

at::Tensor celu(const at::Tensor& self, const at::Scalar& alpha) {
  return NPUCeluFunction::apply(self, alpha);
}

at::Tensor& celu_(at::Tensor& self, const at::Scalar& alpha) {
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguousSelf = npu_utils::format_contiguous(self);
    at::Tensor result = NPUCeluFunction::apply(contiguousSelf, alpha);
    npu_utils::format_fresh_view(self, result);
  } else {
    auto result = NPUCeluFunction::apply(self, alpha);
    self.copy_(result);
  }
  return self;
}
} // namespace op_plugin
