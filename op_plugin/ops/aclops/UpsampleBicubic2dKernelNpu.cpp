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
at::Tensor& upsample_bicubic2d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    TORCH_CHECK(
        output_size.size() == 2,
        "It is expected output_size equals to 2, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    float temp_h = 0.0;
    float temp_w = 0.0;
    if (scales_h.has_value()) {
        temp_h = (float)scales_h.value();
    }
    if (scales_w.has_value()) {
        temp_w = (float)scales_w.value();
    }
    c10::SmallVector<float, SIZE> scales = {temp_h, temp_w};
    c10::SmallVector<float, SIZE> roi = {};
    string coordinate_transformation_mode = "half_pixel";
    if (align_corners == true) {
        coordinate_transformation_mode = "align_corners";
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("ResizeD")
        .Input(self, "X")
        .Output(result, "y")
        .Attr("sizes", output_size)
        .Attr("scales", scales)
        .Attr("roi", roi)
        .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
        .Attr("cubic_coeff_a", (float)-0.75)
        .Attr("exclude_outside", (int64_t)0)
        .Attr("extrapolation_value", (float)0.0)
        .Attr("mode", (string)"cubic")
        .Attr("nearest_mode", (string)"round_prefer_floor")
        .Run();

    return result;
}
} // namespace

at::Tensor& upsample_bicubic2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result)
{
    TORCH_CHECK(self.dim() >= 2, "The self shoud be at least 2D, but self got", self.dim(),
        "D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 2,
        "It is expected output_size equals to 2, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];

    c10::SmallVector<int64_t, SIZE> op_infer_output_size = {N, C, H, W};
    npu_preparation::CheckOut(
        {self},
        result,
        self,
        op_infer_output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        upsample_bicubic2d_out_nocheck(contiguous_result, self, output_size, align_corners, scales_h, scales_w);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        upsample_bicubic2d_out_nocheck(result, self, output_size, align_corners, scales_h, scales_w);
    }

    return result;
}

at::Tensor upsample_bicubic2d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    TORCH_CHECK(self.dim() >= 2, "The self shoud be at least 2D, but self got", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 2,
        "It is expected output_size equals to 2, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    c10::SmallVector<int64_t, SIZE> op_infer_output_size = {N, C, H, W};
    at::Tensor result = npu_preparation::apply_tensor(self, op_infer_output_size);
    upsample_bicubic2d_out_nocheck(result, self, output_size, align_corners, scales_h, scales_w);

    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bicubic2d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        self.dim() == 4,
        "It is expected size equals to 4, but got size ",
        self.dim(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);

    return acl_op::upsample_bicubic2d(self, osize, align_corners, scales_h, scales_w);
}
#endif
} // namespace acl_op
