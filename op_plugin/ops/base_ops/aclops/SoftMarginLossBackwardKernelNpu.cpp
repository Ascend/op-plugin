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
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& soft_margin_loss_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  string reduction_str(op_plugin::utils::get_reduction_str(reduction));

  at_npu::native::OpCommand cmd;
  cmd.Name("SoftMarginLossGrad")
      .Input(input)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reduction_str)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor& soft_margin_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  npu_preparation::CheckOut(
      {grad_output, input, target},
      grad_input,
      input);

  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
    soft_margin_loss_backward_out_nocheck(
        contiguous_grad_input, grad_output, input, target, reduction);
    npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
  } else {
    soft_margin_loss_backward_out_nocheck(
        grad_input, grad_output, input, target, reduction);
  }

  return grad_input;
}

at::Tensor soft_margin_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor grad_input = npu_preparation::apply_tensor(input);
  soft_margin_loss_backward_out_nocheck(
      grad_input, grad_output, input, target, reduction);
  return grad_input;
}
} // namespace acl_op
