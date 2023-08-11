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
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& softplus_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold) {
  at_npu::native::OpCommand cmd;
  cmd.Name("SoftplusV2Grad")
      .Input(grad_output)
      .Input(self)
      .Output(grad_input)
      .Attr("beta", beta)
      .Attr("threshold", threshold)
      .Run();

    return grad_input;
}
} // namespace

at::Tensor& softplus_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  npu_preparation::CheckOut(
      {grad_output, self},
      grad_input,
      self);
  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
    softplus_backward_out_nocheck(contiguous_result, grad_output, self, beta, threshold);
    npu_utils::format_fresh_view(grad_input, contiguous_result);
  } else {
    softplus_backward_out_nocheck(grad_input, grad_output, self, beta, threshold);
  }
  return grad_input;
}

} // namespace op_plugin
