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
at::Tensor& im2col_out_nocheck(
    at::Tensor& result,
    const at::Tensor &self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "im2col: kernel_size must either be a single int, or a tuple of two ints");
  if (kernel_size.size() == 1) {
    c10::SmallVector<int64_t, SIZE> kernel_sizes = {kernel_size[0], kernel_size[0]};
    kernel_size = at::IntArrayRef(kernel_sizes);
  }

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
      "im2col: stride must either be omitted, a single int, or a tuple of two ints");
  stride = stride.empty() ? at::IntArrayRef({1}) : stride;

  TORCH_CHECK(dilation.empty() || dilation.size() == 1 || dilation.size() == 2,
      "im2col: dilation must either be omitted, a single int, or a tuple of two ints");
  dilation = dilation.empty() ? at::IntArrayRef({1}) : dilation;

  TORCH_CHECK(padding.empty() || padding.size() == 1 || padding.size() == 2,
      "im2col: padding must either be omitted, a single int, or a tuple of two ints");

  auto padding_ = padding.empty() ? at::IntArrayRef({0}) : padding;
  c10::SmallVector<int64_t, SIZE> pads;
  if (padding_.size() == 1) {
    pads = {padding_[0], padding_[0], padding_[0], padding_[0]};
  } else if (padding_.size() == 2) {
    pads = {padding_[0], padding_[0], padding_[1], padding_[1]};
  }

  auto padding_4d = at::IntArrayRef(pads);

  int64_t stride_h = 1;
  int64_t stride_w = 1;
  if (stride.size() == 1) {
    stride_h = stride[0];
    stride_w = stride[0];
  } else if (stride.size() == 2) {
    stride_h = stride[0];
    stride_w = stride[1];
  }

  int64_t dilation_h = 1;
  int64_t dilation_w = 1;
  if (dilation.size() == 1) {
    dilation_h = dilation[0];
    dilation_w = dilation[0];
  } else if (dilation.size() == 2) {
    dilation_h = dilation[0];
    dilation_w = dilation[1];
  }

  c10::SmallVector<int64_t, N> kernel_sizes = {kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> stride_sizes = {stride_h, stride_w};
  c10::SmallVector<int64_t, N> dilations_sizes = {dilation_h, dilation_w};
  c10::SmallVector<int64_t, N> pads_size = {padding_4d[0], padding_4d[1], padding_4d[2], padding_4d[3]};
  string padding_mode = "CALCULATED";

  at_npu::native::OpCommand cmd;
  cmd.Name("Im2col")
      .Input(self, "x")
      .Output(result)
      .Attr("ksizes", kernel_sizes)
      .Attr("strides", stride_sizes)
      .Attr("dilations", dilations_sizes)
      .Attr("padding_mode", padding_mode)
      .Attr("pads", pads_size)
      .Run();
  return result;
}
} // namespace

at::Tensor& im2col_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& result) {
  at::Tensor self_cp = self.dim() == 3 ? at::unsqueeze(self, 0) : self;
  auto output_size = op_infer::image_to_col_npu_output_size(self_cp, kernel_size, stride, dilation, padding);

  npu_preparation::CheckOut(
      {self_cp},
      result,
      self_cp,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    im2col_out_nocheck(contiguous_result, self_cp, kernel_size, dilation, padding, stride);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    im2col_out_nocheck(result, self_cp, kernel_size, dilation, padding, stride);
  }

  if (self.dim() == 3) {
    result = at::squeeze(result, 0);
  }
  return result;
}

at::Tensor im2col(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor self_cp = self.dim() == 3 ? at::unsqueeze(self, 0) : self;
  auto output_size = op_infer::image_to_col_npu_output_size(self_cp, kernel_size, stride, dilation, padding);
  at::Tensor result = npu_preparation::ApplyTensor(self_cp, output_size);
  im2col_out_nocheck(result, self_cp, kernel_size, dilation, padding, stride);
  if (self.dim() == 3) {
    result = at::squeeze(result, 0);
  }
  return result;
}
} // namespace acl_op
