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

at::Tensor& reflection_pad2d_backward_out_npu_nocheck(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  c10::SmallVector<int64_t, N> vector_int;
  c10::SmallVector<int64_t, N> paddings_vector = op_infer::array_to_small_vector(padding);
  at::Tensor input_cp = input;
  at::Tensor grad_output_cp = grad_output;
  if (input.dim() == 3) {
    input_cp = input.unsqueeze(0);
    grad_output_cp = grad_output.unsqueeze(0);
    grad_input.unsqueeze_(0);
  }
  paddings_vector.resize(2 * input_cp.dim(), 0);
  for (int64_t i = paddings_vector.size(); i > 0; i -= 2) {
    vector_int.emplace_back(paddings_vector[i - 2]);
    vector_int.emplace_back(paddings_vector[i - 1]);
  }
  at_npu::native::OpCommand cmd;
  cmd.Name("PadV3Grad")
      .Input(grad_output_cp)
      .Input(vector_int, at::kInt)
      .Output(grad_input)
      .Attr("mode", (string)"reflect")
      .Attr("paddings_contiguous", true)
      .Run();

  if (input.dim() == 3) {
    grad_input.squeeze_(0);
  }
  return grad_input;
}
} // namespace

at::Tensor& reflection_pad2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  npu_preparation::CheckOut(
      {input, grad_output},
      grad_input,
      input);
  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
    reflection_pad2d_backward_out_npu_nocheck(grad_output, input, padding, contiguous_result);
    npu_utils::format_fresh_view(grad_input, contiguous_result);
  } else {
    reflection_pad2d_backward_out_npu_nocheck(grad_output, input, padding, grad_input);
  }
  return grad_input;
}

at::Tensor reflection_pad2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef padding) {
  at::Tensor grad_input = npu_preparation::ApplyTensor(input);
  reflection_pad2d_backward_out_npu_nocheck(grad_output, input, padding, grad_input);
  return grad_input;
}

} // namespace op_plugin
