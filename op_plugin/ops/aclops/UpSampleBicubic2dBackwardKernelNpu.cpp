// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
at::Tensor& upsample_bicubic2d_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    TORCH_CHECK(output_size.size() == 2,
        "It is expected output_size equals to 2, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(input_size.size() == 4,
        "It is expected input_size equals to 4, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    float temp_h = 0.0;
    float temp_w = 0.0;
    temp_h = scales_h.has_value() ? static_cast<float>(scales_h.value()) : temp_h;
    temp_w = scales_w.has_value() ? static_cast<float>(scales_w.value()) : temp_w;
    c10::SmallVector<float, N> scales = {temp_h, temp_w};
    c10::SmallVector<float, N> roi = {};
    string coordinate_transformation_mode =
        align_corners ? "align_corners" : "half_pixel";

    float cu = -0.75;
    int64_t ex = 0;
    float ext = 0.0;
    string mode = "cubic";
    string ne = "round_prefer_floor";
    string data_format = "HWNC";
    at_npu::native::OpCommand cmd;
    at::Tensor grad_output_transpose = grad_output.permute({2, 3, 0, 1});

    cmd.Name("ResizeGradD")
      .Input(grad_output_transpose, "grads")
      .Output(grad_input, "y")
      .Attr("scales", scales)
      .Attr("roi", roi)
      .Attr("original_size", input_size)
      .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
      .Attr("cubic_coeff_a", cu)
      .Attr("exclude_outside", ex)
      .Attr("extrapolation_value", ext)
      .Attr("mode", mode)
      .Attr("nearest_mode", ne)
      .Attr("data_format", data_format)
      .Run();
    return grad_input;
}
} // namespace

at::Tensor& upsample_bicubic2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    auto op_infer_output_size = op_infer::upsample_bicubic2d_backward_npu_output_size(input_size);

    npu_preparation::CheckOut(
        {grad_output},
        grad_input,
        grad_output,
        op_infer_output_size);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        upsample_bicubic2d_backward_out_nocheck(
            contiguous_result, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        upsample_bicubic2d_backward_out_nocheck(
            grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }

    int64_t N = grad_output.size(0);
    int64_t C = grad_output.size(1);
    int64_t H = input_size[2];
    int64_t W = input_size[3];

    grad_input = grad_input.reshape({H, W, N, C}).permute({2, 3, 0, 1});
    return grad_input;
}

at::Tensor upsample_bicubic2d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    auto op_infer_output_size = op_infer::upsample_bicubic2d_backward_npu_output_size(input_size);
    at::Tensor result = npu_preparation::apply_tensor(grad_output, op_infer_output_size);
    upsample_bicubic2d_backward_out_nocheck(result, grad_output, output_size, input_size,
        align_corners, scales_h, scales_w);

    int64_t N = grad_output.size(0);
    int64_t C = grad_output.size(1);
    int64_t H = input_size[2];
    int64_t W = input_size[3];
    result = result.reshape({H, W, N, C}).permute({2, 3, 0, 1});
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bicubic2d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input_size.size() == 4,
        "It is expected input_size equals to 4, but got size ",
        input_size.size(),
        OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    return acl_op::upsample_bicubic2d_backward(grad_output, osize, input_size, align_corners, scales_h, scales_w);
}
#endif
} // namespace acl_op
