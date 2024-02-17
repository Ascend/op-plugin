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
at::Tensor& softshrink_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar lambd) {
  at_npu::native::OpCommand cmd;
  cmd.Name("SoftShrinkGrad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("lambd", lambd)
      .Run();

  return grad_input;
}
} // namespace

at::Tensor& softshrink_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& lambd,
    at::Tensor& grad_input) {
  npu_preparation::CheckOut(
      {self, grad_output},
      grad_input,
      self);

  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
    softshrink_backward_out_nocheck(contiguous_result, grad_output, self, lambd);
    npu_utils::format_fresh_view(grad_input, contiguous_result);
  } else {
    softshrink_backward_out_nocheck(grad_input, grad_output, self, lambd);
  }

  return grad_input;
}

at::Tensor softshrink_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& lambd) {
  at::Tensor grad_input = npu_preparation::apply_tensor(self);

  softshrink_backward_out_nocheck(
      grad_input, grad_output, self, lambd);

  return grad_input;
}
} // namespace acl_op
