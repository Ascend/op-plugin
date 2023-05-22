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
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/FormatHelper.h"

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using npu_format_helper = at_npu::native::FormatHelper;
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
at::Tensor linear_npu_nocheck(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor> & bias_opt) {
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  c10::SmallVector<int64_t, SIZE> output_size = {input.size(0), weight.size(0)};
  at::Tensor output = npu_preparation::ApplyTensor(input, output_size);

  int64_t offset_x = 0;
  at_npu::native::OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight);
  if (bias.defined()) {
    cmd.Input(bias);
  }
  cmd.Output(output)
      .Attr("transpose_x1", false)
      .Attr("transpose_x2", true)
      .Attr("offset_x", offset_x)
      .Run();

  return output;
}

at::Tensor linear_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& input,
    const at::Tensor& weight,
    bool transpose_x1,
    bool transpose_x2) {
  int64_t offset_x = 0;
  at_npu::native::OpCommand cmd;
  cmd.Name("MatMulV2")
      .Input(input)
      .Input(weight)
      .Output(result)
      .Attr("transpose_x1", transpose_x1)
      .Attr("transpose_x2", transpose_x2)
      .Attr("offset_x", offset_x)
      .Run();
  return result;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_linear_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& weight) {
  c10::SmallVector<int64_t, SIZE> input_grad_output_size = {
      grad.size(0),
      weight.size(1)};
  c10::SmallVector<int64_t, SIZE> weight_grad_output_size = {
      grad.size(1),
      input.size(1)};
  at::Tensor input_grad = npu_preparation::ApplyTensor(input, input_grad_output_size);
  at::Tensor weight_grad = npu_preparation::ApplyTensor(weight, weight_grad_output_size);

  if (calcu_op_util::GetTensorNpuFormat(grad) == calcu_op_util::GetTensorNpuFormat(weight)) {
    linear_backward_out_npu_nocheck(input_grad, grad, weight, false, false);
    linear_backward_out_npu_nocheck(weight_grad, grad, input, true, false);
  } else {
    at::Tensor gradFormatcast = npu_preparation::ApplyTensor(grad, grad.sizes());
    gradFormatcast =
        at_npu::native::NPUNativeFunctions::npu_format_cast(grad, calcu_op_util::GetTensorNpuFormat(weight));
    linear_backward_out_npu_nocheck(input_grad, gradFormatcast, weight, false, false);
    linear_backward_out_npu_nocheck(weight_grad, gradFormatcast, input, true, false);
  }

  return std::tie(input_grad, weight_grad);
}

class NPULinearFunction : public torch::autograd::Function<NPULinearFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx,
      const at::Tensor& input,
      const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias_opt) {
    ctx->saved_data["bias_has_value"] = (bias_opt.has_value() == true) ? bias_opt.value().requires_grad() : false;

    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({input, weight});
    return linear_npu_nocheck(input, weight, bias_opt);
  }

  static std::vector<at::Tensor> backward(AutogradContext* ctx, std::vector<at::Tensor> grad_outputs) {
    auto bias_has_value = ctx->saved_data["bias_has_value"].toBool();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];

    std::tuple<at::Tensor, at::Tensor> result = op_plugin::npu_linear_backward(grad_outputs[0], input, weight);

    std::vector<at::Tensor> output = {std::get<0>(result), std::get<1>(result), at::Tensor()};
    if (bias_has_value) {
      output = {std::get<0>(result), std::get<1>(result), grad_outputs[0]};
    }
    return output;
  }
};

at::Tensor npu_linear(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt) {
  auto is_aligin = [&]() {
    return (!(static_cast<uint64_t>(input.size(0)) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(input.size(1)) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(weight.size(0)) & 0x0000000F)) &&
           (!(static_cast<uint64_t>(weight.size(1)) & 0x0000000F));
  };

  static auto mm_bmm_nd = !at_npu::native::env::CheckMmBmmNDDisable();
  static bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
  at::Tensor input_cast = (npu_format_helper::IsBaseFormatType(input) && mm_bmm_nd &&
      ((is_support_nd_out && calcu_op_util::IsNdToNzOnTheFly(input, weight)) ||
      (!is_support_nd_out && is_aligin()))) ? input :
      at_npu::native::NPUNativeFunctions::npu_format_cast(input, ACL_FORMAT_FRACTAL_NZ);
  return NPULinearFunction::apply(input_cast, weight, bias_opt);
}
} // namespace op_plugin
