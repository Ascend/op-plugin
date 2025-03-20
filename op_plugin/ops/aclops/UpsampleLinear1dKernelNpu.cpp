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
inline void upsample_linear1d_check(
    const at::Tensor& self,
    at::IntArrayRef output_size)
{
    TORCH_CHECK(
        output_size.size() == 1,
        "It is expected output_size equals to 1, but got size ",
        output_size.size(), OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(
        (self.size(1) != 0 && self.size(2) != 0) && self.dim() == 3,
        "Non-empty 3D data tensor expected but got a tensor with sizes ",
        self.sizes(), OPS_ERROR(ErrCode::PARAM));

    int64_t input_width = self.size(2);
    int64_t output_width = output_size[0];

    TORCH_CHECK(
        input_width > 0 && output_width > 0,
        "Input and output sizes should be greater than 0, but got input (W: ",
        input_width,
        ") and output (W: ",
        output_width,
        ")" + OPS_ERROR(ErrCode::VALUE));
}

at::Tensor& upsample_linear1d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales)
{
    upsample_linear1d_check(self, output_size);
    // Since only NCHW format input is currently supported, first convert the
    // input self (3 dimensions) to 4 dimensions as the input of npu
    at::Tensor selfcp = self.unsqueeze(2);
    TORCH_CHECK(selfcp.size(3) != 0, "selfcp.size(3) == 0." + OPS_ERROR(ErrCode::PARAM));
    // to calculate the value of scale
    c10::SmallVector<float, N> sc = {};
    if (scales.has_value()) {
        sc.push_back(scales.value());
    } else {
        float temp = float(output_size[0]) / float(selfcp.size(3));
        sc.push_back(temp);
    }
    string coordinate_transformation_mode = align_corners ? "align_corners" : "half_pixel";
    string mode = "linear";

    at_npu::native::OpCommand cmd;
    cmd.Name("ResizeD")
        .Input(selfcp, "X")
        .Output(result, "y")
        .Attr("sizes", output_size)
        .Attr("coordinate_transformation_mode", coordinate_transformation_mode)
        .Attr("mode", mode)
        .Attr("scales", sc)
        .Run();

    return result;
}
} // namespace

at::Tensor& upsample_linear1d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& out)
{
    auto output_sizes = op_infer::upsample_linear1d_npu_output_size(
        self, output_size);

    npu_preparation::CheckOut(
        {self},
        out,
        self,
        output_sizes);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        upsample_linear1d_out_nocheck(contiguous_result, self, output_size, align_corners, scales);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        upsample_linear1d_out_nocheck(out, self, output_size, align_corners, scales);
    }

    return out;
}

at::Tensor upsample_linear1d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales)
{
    auto output_sizes = op_infer::upsample_linear1d_npu_output_size(
        self, output_size);
    at::Tensor result = npu_preparation::apply_tensor(self, output_sizes);

    upsample_linear1d_out_nocheck(result, self, output_size, align_corners, scales);

    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_linear1d(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        self.dim() == 3,
        "It is expected input_size equals to 3, but got size ",
        self.dim(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);

    return acl_op::upsample_linear1d(self, osize, align_corners, scales_w);
}
#endif
} // namespace acl_op
