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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> apply_adam_w_out_npu_nocheck(
    at::Tensor& var_out,
    at::Tensor& m_out,
    at::Tensor& v_out,
    at::Scalar beta1_power,
    at::Scalar beta2_power,
    at::Scalar lr,
    at::Scalar weight_decay,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    c10::optional<at::Tensor> max_grad_norm,
    c10::optional<bool> amsgrad,
    c10::optional<bool> maximize) {
  at_npu::native::OpCommand cmd;
  auto var_out_dtype = var_out.scalar_type();
  cmd.Name("ApplyAdamW")
      .Input(var_out)
      .Input(m_out)
      .Input(v_out)
      .Input(beta1_power, var_out_dtype)
      .Input(beta2_power, var_out_dtype)
      .Input(lr, var_out_dtype)
      .Input(weight_decay, var_out_dtype)
      .Input(beta1, var_out_dtype)
      .Input(beta2, var_out_dtype)
      .Input(epsilon, var_out_dtype)
      .Input(grad);
  if (max_grad_norm.has_value()) {
    cmd.Input(max_grad_norm.value());
  } else {
    cmd.Input();
  }
  cmd.Output(var_out)
      .Output(m_out)
      .Output(v_out);
  if (amsgrad != c10::nullopt) {
    cmd.Attr("amsgrad", bool(amsgrad.value())); // at present, the operator supports only false.
  }
  if (maximize != c10::nullopt) {
    cmd.Attr("maximize", bool(maximize.value()));
  }
  cmd.Run();
  return std::tie(var_out, m_out, v_out);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam_w(
    const at::Scalar& beta1_power,
    const at::Scalar& beta2_power,
    const at::Scalar& lr,
    const at::Scalar& weight_decay,
    const at::Scalar& beta1,
    const at::Scalar& beta2,
    const at::Scalar& epsilon,
    const at::Tensor& grad,
    const c10::optional<at::Tensor>& max_grad_norm,
    c10::optional<bool> amsgrad,
    c10::optional<bool> maximize) {
  AT_ERROR("npu_apply_adam_w is not implemented for Tensor");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_w_out(
    const at::Scalar& beta1_power,
    const at::Scalar& beta2_power,
    const at::Scalar& lr,
    const at::Scalar& weight_decay,
    const at::Scalar& beta1,
    const at::Scalar& beta2,
    const at::Scalar& epsilon,
    const at::Tensor& grad,
    const c10::optional<at::Tensor>& max_grad_norm,
    c10::optional<bool> amsgrad,
    c10::optional<bool> maximize,
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v) {
  bool var_match = npu_utils::check_match(&var);
  bool m_match = npu_utils::check_match(&m);
  bool v_match = npu_utils::check_match(&v);

  if ((amsgrad.has_value()) && (amsgrad.value())) {
    TORCH_CHECK(max_grad_norm.has_value(), "if amsgrad is true, max_grad_norm input must be entered");
  }
  if (!(var_match && m_match && v_match)) {
    at::Tensor contiguous_var = var_match ? var : npu_utils::format_contiguous(var);
    at::Tensor contiguous_m = m_match ? m : npu_utils::format_contiguous(m);
    at::Tensor contiguous_v = v_match ? v : npu_utils::format_contiguous(v);
    apply_adam_w_out_npu_nocheck(
        contiguous_var,
        contiguous_m,
        contiguous_v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm,
        amsgrad,
        maximize);
    if (!var_match) {
      npu_utils::format_fresh_view(var, contiguous_var);
    }
    if (!m_match) {
      npu_utils::format_fresh_view(m, contiguous_m);
    }
    if (!v_match) {
      npu_utils::format_fresh_view(v, contiguous_v);
    }
  } else {
    apply_adam_w_out_npu_nocheck(
        var,
        m,
        v,
        beta1_power,
        beta2_power,
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        grad,
        max_grad_norm,
        amsgrad,
        maximize);
  }
  return std::tie(var, m, v);
}
} // namespace op_plugin
