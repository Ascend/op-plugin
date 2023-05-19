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

at::Tensor& avg_pool2d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  if (padding.size() == 1) {
    c10::SmallVector<int64_t, SIZE> paddings = {padding[0], padding[0]};
    padding = at::IntArrayRef(paddings);
  }

  int64_t stride_h = 1;
  int64_t stride_w = 1;
  if (!stride.empty()) {
    stride_h = stride[0];
    stride_w = stride[1];
  }
  c10::SmallVector<int64_t, N> kernel_size_new = {1, 1, kernel_size[0], kernel_size[1]};
  c10::SmallVector<int64_t, N> strides_size_new = {1, 1, stride_h, stride_w};
  c10::SmallVector<int64_t, N> pads = {padding[0], padding[0], padding[1], padding[1]};
  bool exclusive = !count_include_pad;

  at_npu::native::OpCommand cmd;
  cmd.Name("AvgPoolV2")
      .Input(self)
      .Output(result)
      .Attr("ksize", kernel_size_new)
      .Attr("strides", strides_size_new)
      .Attr("padding_mode", (string) "CALCULATED")
      .Attr("pads", pads)
      .Attr("data_format", (string) "NCHW")
      .Attr("global_pooling", false)
      .Attr("ceil_mode", ceil_mode);
  if (self.scalar_type() == at::ScalarType::Half || self.scalar_type() == at::ScalarType::Char) {
    cmd.Attr("exclusive", true);
  } else {
    cmd.Attr("exclusive", exclusive);
  }
  cmd.Run();
  return result;
}

void avg_pool2d_parameter_check(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
      "non-empty 2D or 3D (batch mode) tensor expected for input");
  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");
}
} // namespace

at::Tensor& avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    at::Tensor& result) {
  avg_pool2d_parameter_check(self, kernel_size, stride, padding, divisor_override);

  at::Tensor self_copy = self;
  if (self.dim() == 3) {
    self_copy = self_copy.unsqueeze(0);
  }
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size.size() == 1 ? k_h : kernel_size[1];

  c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

  const int64_t d_h = stride.empty() ? k_h : stride[0];
  const int64_t d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : stride[1];

  c10::SmallVector<int64_t, SIZE> stride_sizes = {d_h, d_w};
  at::IntArrayRef stridess = at::IntArrayRef(stride_sizes);

  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];

  c10::SmallVector<int64_t, SIZE> padding_sizes = {pad_h, pad_w};
  at::IntArrayRef paddingss = at::IntArrayRef(padding_sizes);

  auto output_sizes = op_infer::avg_pool2d_npu_output_size(
      self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);

  npu_preparation::CheckOut(
      {self},
      result,
      self_copy,
      output_sizes);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contig_result = npu_utils::format_contiguous(result);
    avg_pool2d_out_nocheck(
        contig_result, self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
    npu_utils::format_fresh_view(result, contig_result);
  } else {
    avg_pool2d_out_nocheck(
        result, self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
  }

  if (self.dim() == 3) {
    result = result.squeeze(0);
  }
  return result;
}

at::Tensor avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  avg_pool2d_parameter_check(self, kernel_size, stride, padding, divisor_override);

  at::Tensor self_copy = self;
  if (self.dim() == 3) {
    self_copy = self_copy.unsqueeze(0);
  }
  const int64_t k_h = kernel_size[0];
  const int64_t k_w = kernel_size.size() == 1 ? k_h : kernel_size[1];

  c10::SmallVector<int64_t, SIZE> kernel_sizes = {k_h, k_w};
  at::IntArrayRef kernel_sizess = at::IntArrayRef(kernel_sizes);

  const int64_t d_h = stride.empty() ? k_h : stride[0];
  const int64_t d_w = stride.empty() ? k_w : stride.size() == 1 ? d_h : stride[1];

  c10::SmallVector<int64_t, SIZE> stride_sizes = {d_h, d_w};
  at::IntArrayRef stridess = at::IntArrayRef(stride_sizes);

  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];

  c10::SmallVector<int64_t, SIZE> padding_sizes = {pad_h, pad_w};
  at::IntArrayRef paddingss = at::IntArrayRef(padding_sizes);

  auto output_sizes = op_infer::avg_pool2d_npu_output_size(
      self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
  at::Tensor result = npu_preparation::ApplyTensor(self_copy, output_sizes);

  avg_pool2d_out_nocheck(
      result, self_copy, kernel_sizess, stridess, paddingss, ceil_mode, count_include_pad, divisor_override);
  if (self.dim() == 3) {
    result = result.squeeze(0);
  }
  return result;
}

} // namespace op_plugin
