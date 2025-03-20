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
at::SmallVector<int64_t, SIZE> upsample_nearest2d_infer_size(const at::Tensor &input, at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() == 4, "The input should be 4D, but got ", input.dim(), "D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 2, "The length of output_size should be equal to 2, but got ",
                output_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = output_size[0];
    int64_t W = output_size[1];
    at::SmallVector<int64_t, SIZE> output_sizes = {N, C, H, W};

    return output_sizes;
}

at::Tensor &upsample_nearest2d_out_nocheck(at::Tensor &result, const at::Tensor &self, at::IntArrayRef output_size,
                                           c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    at::SmallVector<int64_t, N> output_size_vec = op_infer::array_to_small_vector(output_size);

    at_npu::native::OpCommand cmd;
    cmd.Name("ResizeNearestNeighborV2")
        .Input(self, "x")
        .Input(output_size_vec, at::kInt)
        .Output(result, "y")
        .Attr("align_corners", false)
        .Attr("half_pixel_centers", false)
        .Run();

    return result;
}
} // namespace

at::Tensor &upsample_nearest2d_out(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales_h,
                                   c10::optional<double> scales_w, at::Tensor &result)
{
    at::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest2d_infer_size(self, output_size);
    npu_preparation::CheckOut({self}, result, npu_preparation::get_tensor_npu_format(result), self.scalar_type(),
                              op_infer_output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        upsample_nearest2d_out_nocheck(contiguous_result, self, output_size, scales_h, scales_w);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        upsample_nearest2d_out_nocheck(result, self, output_size, scales_h, scales_w);
    }

    return result;
}

at::Tensor upsample_nearest2d(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales_h,
                              c10::optional<double> scales_w)
{
    at::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest2d_infer_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor(self, op_infer_output_size);
    upsample_nearest2d_out_nocheck(result, self, output_size, scales_h, scales_w);

    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest2d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input.dim() == 4,
        "It is expected size equals to 4, but got size ",
        input.dim(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
    auto scale_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scale_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    return acl_op::upsample_nearest2d(input, osize, scale_h, scale_w);
}
#endif

} // namespace acl_op
