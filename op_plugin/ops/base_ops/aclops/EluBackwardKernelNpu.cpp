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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& elu_backward_out_nocheck(
    at::Tensor & result,
    const at::Tensor & grad_output,
    const at::Tensor & self_or_result,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale,
    bool is_result) {
  float alphaValue = op_plugin::utils::get_scalar_float_value(alpha);
  float scaleValue = op_plugin::utils::get_scalar_float_value(scale);
  float inputScaleValue = op_plugin::utils::get_scalar_float_value(input_scale);

  if (is_result) {
    TORCH_CHECK((alphaValue >= 0),
        "In-place elu backward calculation is triggered with a negative slope which is not supported. "
        "This is caused by calling in-place forward function with a negative slope, "
        "please call out-of-place version instead." + OPS_ERROR(ErrCode::VALUE));
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("EluGradV2")
      .Input(grad_output)
      .Input(self_or_result)
      .Output(result)
      .Attr("alpha", alphaValue)
      .Attr("scale", scaleValue)
      .Attr("input_scale", inputScaleValue)
      .Attr("is_result", is_result)
      .Run();
  return result;
}
} // namespace

at::Tensor& elu_backward_out(
    const at::Tensor& grad_output,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    bool is_result,
    const at::Tensor& self_or_result,
    at::Tensor & grad_input) {
  elu_backward_out_nocheck(grad_input, grad_output, self_or_result, alpha, scale, input_scale, is_result);
  return grad_input;
}

at::Tensor elu_backward(
    const at::Tensor& grad_output,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    bool is_result,
    const at::Tensor& self_or_result) {
  at::Tensor result = npu_preparation::apply_tensor(grad_output);
  result = elu_backward_out_nocheck(result, grad_output, self_or_result, alpha, scale, input_scale, is_result);
  return result;
}
} // namespace acl_op
