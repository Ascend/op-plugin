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
at::SmallVector<int64_t, SIZE> upsample_trilinear3d_backward_infer_size(
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    TORCH_CHECK(
        output_size.size() == 3,
        "It is expected output_size equals to 3, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(
        input_size.size() == 5,
        "It is expected input_size equals to 5, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t nbatch = input_size[0];
    int64_t channels = input_size[1];
    int64_t input_depth = input_size[2];
    int64_t input_height = input_size[3];
    int64_t input_width = input_size[4];

    at::SmallVector<int64_t, SIZE> output_sizes =
    {nbatch, channels, input_depth, input_height, input_width};
    return output_sizes;
}

at::Tensor& upsample_trilinear3d_backward_out_nocheck(
    at::Tensor& out,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("UpsampleTrilinear3dGrad")
        .Input(grad_output)
        .Output(out)
        .Attr("input_size", input_size)
        .Attr("output_size", output_size)
        .Attr("align_corners", align_corners)
        .Run();

    return out;
}
} // namespace

at::Tensor& upsample_trilinear3d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    auto op_infer_output_size = upsample_trilinear3d_backward_infer_size(
        output_size, input_size, scales_d, scales_h, scales_w);
    npu_preparation::CheckOut(
        {grad_output},
        grad_input,
        grad_output,
        op_infer_output_size);

    if (!npu_utils::check_match(&grad_input)) {
        auto contiguous_out = npu_utils::format_contiguous(grad_input);
        upsample_trilinear3d_backward_out_nocheck(
            grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
        npu_utils::format_fresh_view(grad_input, contiguous_out);
    } else {
        upsample_trilinear3d_backward_out_nocheck(
            grad_input, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    }
    return grad_input;
}

at::Tensor upsample_trilinear3d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    auto op_infer_output_size = upsample_trilinear3d_backward_infer_size(
        output_size, input_size, scales_d, scales_h, scales_w);
    at::Tensor result = npu_preparation::apply_tensor(grad_output, op_infer_output_size);
    upsample_trilinear3d_backward_out_nocheck(
        result, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_trilinear3d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input_size.size() == 5,
        "It is expected input_size equals to 5, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_d = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 1);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 2);

    return acl_op::upsample_trilinear3d_backward(
        grad_output, osize, input_size, align_corners, scales_d, scales_h, scales_w);
}
#endif

} // namespace acl_op
