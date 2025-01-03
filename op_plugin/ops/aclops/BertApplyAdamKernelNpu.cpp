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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> bert_apply_adam_out_npu_nocheck(
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v,
    at::Scalar lr,
    at::Scalar beta1,
    at::Scalar beta2,
    at::Scalar epsilon,
    const at::Tensor& grad,
    at::Scalar max_grad_norm,
    at::Scalar global_grad_norm,
    at::Scalar weight_decay,
    c10::optional<at::Scalar> step_size,
    int64_t adam_mode) {
  std::string adamMode = adam_mode == 0 ? "adam" : "mbart_adam";
  at_npu::native::OpCommand cmd;
  cmd.Name("ApplyAdamV2")
      .Input(var)
      .Input(m)
      .Input(v)
      .Input(lr, var.scalar_type())
      .Input(beta1, var.scalar_type())
      .Input(beta2, var.scalar_type())
      .Input(epsilon, var.scalar_type())
      .Input(grad)
      .Input(max_grad_norm, var.scalar_type())
      .Input(global_grad_norm, var.scalar_type())
      .Input(weight_decay, var.scalar_type());
  if (step_size.has_value()) {
    cmd.Input(step_size.value(), var.scalar_type());
  }
  cmd.Output(var)
      .Output(m)
      .Output(v)
      .Attr("adam_mode", adamMode)
      .Run();
  return std::tie(var, m, v);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_bert_apply_adam(
    const c10::Scalar& lr,
    const c10::Scalar& beta1,
    const c10::Scalar& beta2,
    const c10::Scalar& epsilon,
    const at::Tensor& grad,
    const c10::Scalar& max_grad_norm,
    const c10::Scalar& global_grad_norm,
    const c10::Scalar& weight_decay,
    const c10::optional<at::Scalar>& step_size,
    int64_t adam_mode) {
  TORCH_CHECK(false, "npu_bert_apply_adam is not implemented for Tensor"
      + OPS_ERROR(ErrCode::PARAM));
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_bert_apply_adam_out(
    const c10::Scalar& lr,
    const c10::Scalar& beta1,
    const c10::Scalar& beta2,
    const c10::Scalar& epsilon,
    const at::Tensor& grad,
    const c10::Scalar& max_grad_norm,
    const c10::Scalar& global_grad_norm,
    const c10::Scalar& weight_decay,
    const c10::optional<at::Scalar>& step_size,
    int64_t adam_mode,
    at::Tensor& var,
    at::Tensor& m,
    at::Tensor& v) {
  bool var_match = npu_utils::check_match(&var);
  bool m_match = npu_utils::check_match(&m);
  bool v_match = npu_utils::check_match(&v);
  if (!(var_match && m_match && v_match)) {
    at::Tensor contiguous_var = var_match ? var : npu_utils::format_contiguous(var);
    at::Tensor contiguous_m = m_match ? m : npu_utils::format_contiguous(m);
    at::Tensor contiguous_v = v_match ? v : npu_utils::format_contiguous(v);
    bert_apply_adam_out_npu_nocheck(
        contiguous_var, contiguous_m, contiguous_v, lr, beta1, beta2, epsilon, grad,
        max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
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
    bert_apply_adam_out_npu_nocheck(
        var, m, v, lr, beta1, beta2, epsilon, grad,
        max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
  }
  return std::tie(var, m, v);
}
} // namespace acl_op
