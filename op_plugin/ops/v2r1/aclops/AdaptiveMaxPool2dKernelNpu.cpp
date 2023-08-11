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
inline void adaptive_max_pool2d_check(const at::Tensor& self, at::IntArrayRef output_size) {
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_max_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }
  TORCH_CHECK(
      (self.dim() == 3 || self.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");
  TORCH_CHECK(
      (output_size.size() == 2),
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");
}

std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> adaptive_max_pool2d_infer_size(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  int64_t n = self.size(0);
  int64_t c = self.size(1);
  int64_t h = self.size(2);
  int64_t w = self.size(3);
  int64_t stride_h = h / output_size[0];
  int64_t stride_w = w / output_size[1];
  int64_t kernel_size_h = h - (output_size[0] - 1) * stride_h;
  int64_t kernel_size_w = w - (output_size[1] - 1) * stride_w;
  int64_t Ho = output_size[0];
  int64_t Wo = output_size[1];
  c10::SmallVector<int64_t, SIZE> output_sizes = {n, c, Ho, Wo};
  const int64_t BLOCKSIZE = 16;
  int64_t mask_h = kernel_size_h * kernel_size_w;
  int64_t mask_w = (op_infer::CeilDiv(Ho * Wo, BLOCKSIZE) + 1);
  c10::SmallVector<int64_t, SIZE> indices_size = {n, c, mask_h, mask_w};

  return std::tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(output_sizes, indices_size);
}

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool2d_out_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  npu_preparation::CheckMemory({self}, {output, indices});
  auto inputsize = self.sizes();
  c10::SmallVector<int64_t, N> input_size;
  if (inputsize.size() == 3) {
    c10::SmallVector<int64_t, N> size = { inputsize[1], inputsize[2] };
    input_size = at::IntArrayRef(size);
  } else if (inputsize.size() == 4) {
    c10::SmallVector<int64_t, N> size = { inputsize[2], inputsize[3] };
    input_size = at::IntArrayRef(size);
  }

  // H and W can not be divided, temporarily reported error processing
  TORCH_CHECK(input_size[0] % output_size[0] == 0 && input_size[1] % output_size[1] == 0,
      "H and W must be divisible");

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
  c10::SmallVector<int64_t, N> kernelSize = {1, kernel_size[0], kernel_size[1], 1};
  c10::SmallVector<int64_t, N> stridesSize = {1, stride[0], stride[1], 1};
  c10::SmallVector<int64_t, N> paddings = {1, padding[0], padding[1], 1};
  c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};
  bool ceil_mode = false;

  at_npu::native::OpCommand cmd;
  cmd.Name("MaxPoolWithArgmaxV1")
      .Input(self, "x")
      .Output(output, "y")
      .Output(indices, "argmax", c10::nullopt, "uint16")
      .Attr("ksize", kernelSize)
      .Attr("strides", stridesSize)
      .Attr("pads", paddings)
      .Attr("dilation", dilations)
      .Attr("ceil_mode", ceil_mode)
      .Run();

  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> adaptive_max_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& output,
    at::Tensor& indices) {
  adaptive_max_pool2d_check(self, output_size);
  c10::SmallVector<int64_t, SIZE> output_sizes = std::get<0>(adaptive_max_pool2d_infer_size(self, output_size));
  c10::SmallVector<int64_t, SIZE> indices_size = std::get<1>(adaptive_max_pool2d_infer_size(self, output_size));

  npu_preparation::CheckOut(
      {self},
      output,
      self,
      output_sizes);
  npu_preparation::CheckOut(
      {self},
      indices,
      ACL_FORMAT_NC1HWC0,
      at::ScalarType::Long,
      indices_size);

  bool output_match = npu_utils::check_match(&output);
  bool indices_match = npu_utils::check_match(&indices);
  if (!(output_match && indices_match)) {
    at::Tensor contiguous_output = output_match ? output : npu_utils::format_contiguous(output);
    at::Tensor contiguous_indices = indices_match ? indices : npu_utils::format_contiguous(indices);
    adaptive_max_pool2d_out_nocheck(contiguous_output, contiguous_indices, self, output_size);
    if (!output_match) {
      npu_utils::format_fresh_view(output, contiguous_output);
    }
    if (!indices_match) {
      npu_utils::format_fresh_view(indices, contiguous_indices);
    }
  } else {
    adaptive_max_pool2d_out_nocheck(output, indices, self, output_size);
  }

  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool2d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  adaptive_max_pool2d_check(self, output_size);
  c10::SmallVector<int64_t, SIZE> output_sizes = std::get<0>(adaptive_max_pool2d_infer_size(self, output_size));
  c10::SmallVector<int64_t, SIZE> indices_size = std::get<1>(adaptive_max_pool2d_infer_size(self, output_size));
  at::Tensor output = npu_preparation::ApplyTensor(self, output_sizes);
  at::Tensor indices = npu_preparation::ApplyTensorWithFormat(indices_size, self.options().dtype(at::kLong), ACL_FORMAT_NC1HWC0);
  adaptive_max_pool2d_out_nocheck(output, indices, self, output_size);

  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}
} // namespace op_plugin
