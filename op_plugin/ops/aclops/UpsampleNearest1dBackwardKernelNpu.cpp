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
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_backward_infer_size(at::IntArrayRef input_size)
{
    TORCH_CHECK(
        input_size.size() == 3,
        "It is expected input_size equals to 3, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, SIZE> output_size;
    int64_t N = input_size[0];
    int64_t C = input_size[1];
    int64_t W = input_size[2];
    output_size = {N, C, 1, W};
    return output_size;
}

at::Tensor& upsample_nearest1d_backward_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales)
{
    at::Tensor grad_cp = grad_output.unsqueeze(2);
    at_npu::native::OpCommand cmd;
    if (grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kHalf) {
        c10::SmallVector<int64_t, SIZE> result_size = {1, input_size[2]};
        cmd.Name("ResizeNearestNeighborV2Grad")
            .Input(grad_cp)
            .Input(result_size, at::kInt)
            .Output(grad_input)
            .Attr("align_corners", false)
            .Attr("half_pixel_centers", false)
            .Run();
    } else {
        TORCH_CHECK(output_size[0] != 0, "output_size must not equals to 0, but got ", output_size[0],
            OPS_ERROR(ErrCode::PARAM));
        c10::SmallVector<int64_t, SIZE> origin_size = upsample_nearest1d_backward_infer_size(input_size);
        at::Scalar scales_cp = scales.has_value() ? scales.value() : -1;
        cmd.Name("ResizeGrad")
            .Input(grad_cp)
            .Input(scales_cp, at::kFloat)
            .Input(scales_cp, at::kFloat)
            .Input(origin_size, at::kLong)
            .Output(grad_input)
            // Default value of Resize
            .Attr("coordinate_transformation_mode", (string)"pytorch_half_pixel")
            .Attr("cubic_coeff_a", (float)-0.75)
            .Attr("exclude_outside", (int64_t)0)
            .Attr("extrapolation_value", (float)0.0)
            .Attr("mode", (string)"nearest")
            .Attr("nearest_mode", (string)"floor")
            .Run();
    }
    grad_input = grad_input.squeeze(2);
    return grad_input;
}
} // namespace

at::Tensor& upsample_nearest1d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input)
{
    c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_backward_infer_size(input_size);
    npu_preparation::CheckOut(
        {grad_output},
        grad_input,
        grad_output,
        op_infer_output_size);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        upsample_nearest1d_backward_out_nocheck(contiguous_result, grad_output, output_size, input_size, scales);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        upsample_nearest1d_backward_out_nocheck(grad_input, grad_output, output_size, input_size, scales);
    }

    return grad_input;
}

at::Tensor upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales)
{
    c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_backward_infer_size(input_size);
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, op_infer_output_size);

    upsample_nearest1d_backward_out_nocheck(
        grad_input, grad_output, output_size, input_size, scales);
    return grad_input;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input_size.size() == 3,
        "It is expected input_size equals to 3, but got size ",
        input_size.size(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);

    return acl_op::upsample_nearest1d_backward(grad_output, osize, input_size, scales_w);
}
#endif
} // namespace acl_op
