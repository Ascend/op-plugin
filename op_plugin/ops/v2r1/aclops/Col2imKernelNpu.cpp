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
at::Tensor& col2im_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor grad_output_cp = grad_output;
  grad_output_cp = grad_output_cp.view({
      grad_output.size(0),
      grad_output.size(1) / (kernel_size[0] * kernel_size[1]),
      kernel_size[0] * kernel_size[1],
      grad_output.size(2)});
  c10::SmallVector<int64_t, N> input_sizes = {input_size[0], input_size[1]};
  c10::SmallVector<int64_t, N> kernel_sizes = {kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> dilations = {dilation[0], dilation[1]};
  c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1]};
  c10::SmallVector<int64_t, N> strides_sizes = {stride[0], stride[1]};
  at_npu::native::OpCommand cmd;
  cmd.Name("Col2im")
      .Input(grad_output_cp, "x")
      .Input(input_sizes, at::kInt)
      .Output(grad_input, "y")
      .Attr("kernel_size", kernel_sizes)
      .Attr("dilation", dilations)
      .Attr("padding", paddings)
      .Attr("stride", strides_sizes)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor& col2im_out(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& grad_input) {
  at::Tensor grad_output_cp = grad_output.dim() == 2 ? at::unsqueeze(grad_output, 0) : grad_output;
  c10::SmallVector<int64_t, SIZE> output_size = {
    grad_output_cp.size(0),
    grad_output_cp.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]
  };

  npu_preparation::CheckOut(
      {grad_output_cp},
      grad_input,
      grad_output_cp,
      output_size);

  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
    col2im_out_npu_nocheck(contiguous_grad_input, grad_output_cp, input_size, kernel_size, dilation, padding, stride);
    npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
  } else {
    col2im_out_npu_nocheck(grad_input, grad_output_cp, input_size, kernel_size, dilation, padding, stride);
  }

  if (grad_output.dim() == 2) {
    grad_input = at::squeeze(grad_input, 0);
  }
  return grad_input;
}

at::Tensor col2im(
    const at::Tensor& grad_output,
    at::IntArrayRef input_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor grad_output_cp = grad_output.dim() == 2 ? at::unsqueeze(grad_output, 0) : grad_output;
  c10::SmallVector<int64_t, SIZE> output_size = {
    grad_output_cp.size(0),
    grad_output_cp.size(1) / (kernel_size[0] * kernel_size[1]),
    input_size[0],
    input_size[1]
    };

  at::Tensor grad_input = npu_preparation::ApplyTensor(grad_output_cp, output_size);
  col2im_out_npu_nocheck(grad_input, grad_output_cp, input_size, kernel_size, dilation, padding, stride);

  if (grad_output.dim() == 2) {
    grad_input = at::squeeze(grad_input, 0);
  }
  return grad_input;
}
} // namespace op_plugin
