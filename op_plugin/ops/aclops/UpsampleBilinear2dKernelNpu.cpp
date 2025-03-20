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
at::Tensor& upsample_bilinear2d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    at_npu::native::OpCommand cmd;
    bool half_pixel_centers = !align_corners;
    TORCH_CHECK(output_size.size() >= 1, "The dim input tensor [output_size] must more than 1."
        + OPS_ERROR(ErrCode::PARAM));
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    at::SmallVector<int64_t, N> attr_size = {H, W};
    cmd.Name("ResizeBilinearV2")
        .Input(self, "x")
        .Input(attr_size, at::kInt)
        .Output(result, "y")
        .Attr("align_corners", align_corners)
        .Attr("half_pixel_centers", half_pixel_centers)
        .Run();
    return result;
}
} // namespace

at::Tensor& upsample_bilinear2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result)
{
    at::Tensor self_apply = self;
    if (self_apply.scalar_type() != at::ScalarType::Float) {
        self_apply = at_npu::native::custom_ops::npu_dtype_cast(self_apply, at::ScalarType::Float);
    }
    auto op_infer_output_size = op_infer::upsample_bilinear2d_npu_output_size(
        self_apply, output_size);

    npu_preparation::CheckOut(
        {self_apply},
        result,
        self_apply,
        op_infer_output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        upsample_bilinear2d_out_nocheck(
            contiguous_result, self_apply, output_size, align_corners, scales_h, scales_w);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        upsample_bilinear2d_out_nocheck(
            result, self_apply, output_size, align_corners, scales_h, scales_w);
    }
    return result;
}

at::Tensor upsample_bilinear2d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    at::Tensor self_apply = self;
    if (self_apply.scalar_type() != at::ScalarType::Float) {
        self_apply = at_npu::native::custom_ops::npu_dtype_cast(self_apply, at::ScalarType::Float);
    }
    auto op_infer_output_size = op_infer::upsample_bilinear2d_npu_output_size(
        self_apply, output_size);
    at::Tensor result = npu_preparation::apply_tensor(self_apply, op_infer_output_size);

    upsample_bilinear2d_out_nocheck(
        result, self_apply, output_size, align_corners, scales_h, scales_w);
    if (result.dtype() != self.dtype()) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, self.scalar_type());
    }
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bilinear2d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        self.dim() == 4,
        "It is expected size equals to 4, but got size ",
        self.dim(),
        OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);

    return acl_op::upsample_bilinear2d(self, osize, align_corners, scales_h, scales_w);
}
#endif
} // namespace acl_op
