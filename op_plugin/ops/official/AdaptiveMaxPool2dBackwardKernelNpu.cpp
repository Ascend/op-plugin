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
at::Tensor& adaptive_max_pool2d_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices) {
  auto inputsize = self.sizes();
  c10::SmallVector<int64_t, N> input_size;
  if (inputsize.size() == 3) {
    c10::SmallVector<int64_t, N> size = { inputsize[1], inputsize[2] };
    input_size = at::IntArrayRef(size);
  } else if (inputsize.size() == 4) {
    c10::SmallVector<int64_t, N> size = { inputsize[2], inputsize[3] };
    input_size = at::IntArrayRef(size);
  }
  c10::SmallVector<int64_t, N> output_size = {
    grad_output.size(grad_output.dim() - 2),
    grad_output.size(grad_output.dim() - 1)};

  // H and W can not be divided, temporarily reported error processing
  TORCH_CHECK(input_size[0] % output_size[0] == 0 && input_size[1] % output_size[1] == 0,
      "H and W must be divisible.");
  int64_t kernel_size[2];
  int64_t stride[2];
  int64_t padding[2];
  int64_t stride_h = input_size[0] / output_size[0];
  int64_t stride_w = input_size[1] / output_size[1];
  int64_t kernel_size_h = input_size[0] - (output_size[0] - 1) * stride_h;
  int64_t kernel_size_w = input_size[1] - (output_size[1] - 1) * stride_w;
  stride[0] = stride_h;
  stride[1] = stride_w;
  kernel_size[0] = kernel_size_h;
  kernel_size[1] = kernel_size_w;
  padding[0] = padding[1] = 0;
  c10::SmallVector<int64_t, N> kernel_sizes = {1, kernel_size[0], kernel_size[1], 1};
  c10::SmallVector<int64_t, N> strides_size = {1, stride[0], stride[1], 1};
  c10::SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
  bool ceil_mode = false;
  at_npu::native::OpCommand cmd;
  cmd.Name("MaxPoolGradWithArgmaxV1")
      .Input(self, "x", ACL_FORMAT_NCHW)
      .Input(grad_output, "grad", ACL_FORMAT_NCHW)
      .Input(indices, "argmax", ACL_FORMAT_NCHW, "uint16")
      .Output(grad_input, "y", ACL_FORMAT_NCHW)
      .Attr("ksize", kernel_sizes)
      .Attr("strides", strides_size)
      .Attr("pads", paddings)
      .Attr("dilations", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();

  return grad_input;
}
} // namespace

at::Tensor& adaptive_max_pool2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::Tensor& grad_input) {
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  npu_preparation::CheckOut(
      {grad_output, self, indices},
      grad_input,
      ACL_FORMAT_NC1HWC0,
      self.scalar_type(),
      self.sizes());
  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
    adaptive_max_pool2d_backward_out_nocheck(contiguous_grad_input, grad_output, self, indices);
    npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
  } else {
    adaptive_max_pool2d_backward_out_nocheck(grad_input, grad_output, self, indices);
  }

  return grad_input;
}

at::Tensor adaptive_max_pool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices) {
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  at::Tensor grad_input = npu_preparation::ApplyTensorWithFormat(self, ACL_FORMAT_NC1HWC0);
  adaptive_max_pool2d_backward_out_nocheck(grad_input, grad_output, self, indices);
  return grad_input;
}
} // namespace op_plugin
