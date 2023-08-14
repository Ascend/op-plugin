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
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_infer_size(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  c10::SmallVector<int64_t, SIZE> output_sizes;
  int64_t N = input.size(0);
  int64_t C = input.size(1);
  int64_t W = output_size[0];
  output_sizes = {N, C, 1, W};
  return output_sizes;
}

at::Tensor& upsample_nearest1d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
  TORCH_CHECK(
      (self.size(1) != 0 && self.size(2) != 0) && self.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      self.sizes());

  at::Tensor self_cp = self.unsqueeze(2);
  at_npu::native::OpCommand cmd;
  if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf) {
    c10::SmallVector<int64_t, SIZE> result_size = {1, output_size[0]};
    cmd.Name("ResizeNearestNeighborV2")
        .Input(self_cp)
        .Input(result_size, at::kInt)
        .Output(result)
        .Attr("align_corners", false)
        .Attr("half_pixel_centers", false)
        .Run();
  } else {
    cmd.Name("Resize")
        .Input(self_cp)
        .Input(output_size, at::kFloat)
        .Input(output_size, at::kFloat)
        .Input(result.sizes(), at::kLong)
        .Output(result)
        .Attr("mode", (string)"nearest")
        .Attr("nearest_mode", (string)"floor")
        .Attr("coordinate_transformation_mode", (string)"pytorch_half_pixel")
        .Run();
  }
  result = result.squeeze(2);
  return result;
}
} // namespace

at::Tensor& upsample_nearest1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& result) {
  c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_infer_size(self, output_size);

  npu_preparation::CheckOut(
      {self},
      result,
      self,
      op_infer_output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    upsample_nearest1d_out_nocheck(contiguous_result, self, output_size, scales);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    upsample_nearest1d_out_nocheck(result, self, output_size, scales);
  }

  return result;
}

at::Tensor upsample_nearest1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales) {
  c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_infer_size(self, output_size);
  at::Tensor result = npu_preparation::apply_tensor(self, op_infer_output_size);

  upsample_nearest1d_out_nocheck(result, self, output_size, scales);
  return result;
}
} // namespace op_plugin
