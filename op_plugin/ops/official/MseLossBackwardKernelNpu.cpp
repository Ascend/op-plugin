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
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& mse_loss_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    string reduction_str) {
  at_npu::native::OpCommand cmd;
  cmd.Name("MseLossGrad")
      .Input(self)
      .Input(target)
      .Input(grad_output)
      .Output(grad_input)
      .Attr("reduction", reduction_str)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor& mse_loss_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& grad_input) {
  if (self.numel() == 0 || target.numel() == 0) {
    grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    return grad_input;
  }

  npu_preparation::CheckOut(
      {grad_output, self, target},
      grad_input,
      self);

  string reduction_str(calcu_op_util::GetReductionStr(reduction));

  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
    mse_loss_backward_out_npu_nocheck(contiguous_grad_input, grad_output, self, target, reduction_str);
    npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
  } else {
    mse_loss_backward_out_npu_nocheck(grad_input, grad_output, self, target, reduction_str);
  }

  return grad_input;
}

at::Tensor mse_loss_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  auto grad_out = grad_output.contiguous();
  if (grad_out.dim() == 0) {
    grad_out.view(1);
  }
  at::Tensor grad_input = npu_preparation::ApplyTensor(self);
  op_plugin::mse_loss_backward_out(
      grad_out,
      self,
      target,
      reduction,
      grad_input);
  return grad_input;
}
} // namespace op_plugin
