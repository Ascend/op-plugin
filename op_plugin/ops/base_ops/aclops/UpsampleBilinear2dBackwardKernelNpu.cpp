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
at::Tensor& upsample_bilinear2d_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  at_npu::native::OpCommand cmd;
  at::Tensor original_image = npu_preparation::apply_tensor(grad_output, input_size);
  bool half_pixel_centers = !align_corners;
  cmd.Name("ResizeBilinearV2Grad")
      .Input(grad_output, "grads")
      .Input(original_image, "original_image")
      .Output(grad_input, "y")
      .Attr("align_corners", align_corners)
      .Attr("half_pixel_centers", half_pixel_centers)
      .Run();
  return grad_input;
}
} // namespace

at::Tensor& upsample_bilinear2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  npu_preparation::CheckOut(
      {grad_output},
      grad_input,
      grad_output,
      input_size);
  if (!npu_utils::check_match(&grad_input)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
    upsample_bilinear2d_backward_out_nocheck(
        contiguous_result, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    npu_utils::format_fresh_view(grad_input, contiguous_result);
  } else {
    upsample_bilinear2d_backward_out_nocheck(
        grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  }
  return grad_input;
}

at::Tensor upsample_bilinear2d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto op_infer_output_size = op_infer::upsample_bilinear2d_backward_npu_output_size(
      grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, op_infer_output_size);

  upsample_bilinear2d_backward_out_nocheck(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  return grad_input;
}
} // namespace op_plugin
