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
at::Tensor& elu_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
  float alpha_value = op_plugin::utils::get_scalar_float_value(alpha);
  float scale_value = op_plugin::utils::get_scalar_float_value(scale);
  float input_scale_value = op_plugin::utils::get_scalar_float_value(input_scale);
  at_npu::native::OpCommand cmd;
  cmd.Name("Elu")
      .Input(self)
      .Output(result)
      .Attr("alpha", alpha_value)
      .Attr("scale", scale_value)
      .Attr("input_scale", input_scale_value)
      .Run();
  return result;
}

at::Tensor elu_npu_impl(
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  elu_out_nocheck(result, self, alpha, scale, input_scale);
  return result;
}

at::Tensor& elu_backward_out_npu(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const at::Tensor& output) {
  float value = op_plugin::utils::get_scalar_float_value(alpha);
  at_npu::native::OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(output)
      .Output(grad_input)
      .Attr("alpha", value)
      .Run();
  return grad_input;
}
at::Tensor elu_backward_npu_impl(
    const at::Tensor& grad_output,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    const at::Tensor& output) {
  at::Tensor result = npu_preparation::ApplyTensor(grad_output);
  elu_backward_out_npu(result, grad_output, alpha, scale, input_scale, output);
  return result;
}
} // namespace

at::Tensor& elu_out(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    elu_out_nocheck(contiguous_result, self, alpha, scale, input_scale);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    elu_out_nocheck(result, self, alpha, scale, input_scale);
  }
  return result;
}

class NPUEluFunction: public torch::autograd::Function<NPUEluFunction> {
public:
  static at::Tensor forward(AutogradContext *ctx,
      const at::Tensor& self, 
      at::Scalar alpha, 
      at::Scalar scale, 
      at::Scalar input_scale) {
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["input_scale"] = input_scale;
    at::AutoNonVariableTypeMode g;
    at::Tensor result = elu_npu_impl(self, alpha, scale, input_scale);
    ctx->save_for_backward({result});
    return result;
  }

  static std::vector<at::Tensor> backward(AutogradContext *ctx,
      std::vector<at::Tensor> grad_outputs) {
    auto alpha = ctx->saved_data["alpha"].toScalar();
    auto scale = ctx->saved_data["scale"].toScalar();
    auto input_scale = ctx->saved_data["input_scale"].toScalar();
    auto saved = ctx->get_saved_variables();
    auto result = saved[0];
    auto grad_input = elu_backward_npu_impl(
        grad_outputs[0], 
        alpha,
        scale,
        input_scale, 
        result);
    std::vector<at::Tensor> output = {grad_input, at::Tensor(), at::Tensor(), at::Tensor()};
    return output;
  }
};

at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
  return NPUEluFunction::apply(self, alpha, scale, input_scale);
}

at::Tensor& elu_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
  auto result = NPUEluFunction::apply(self, alpha, scale, input_scale);
  self.copy_(result);
  return self;
}
} // namespace op_plugin
