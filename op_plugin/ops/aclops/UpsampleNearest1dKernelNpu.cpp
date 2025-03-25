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
c10::SmallVector<int64_t, SIZE> upsample_nearest1d_infer_size(const at::Tensor &input, at::IntArrayRef output_size)
{
    TORCH_CHECK(input.dim() == 3 && (input.size(1) != 0 && input.size(2) != 0),
                "Non-empty 3D data tensor expected but got a tensor with sizes ", input.sizes(),
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(output_size.size() == 1, "The length of output_size should be equal to 1, but got ",
                output_size.size(), OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, SIZE> output_sizes;
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t W = output_size[0];
    output_sizes = {N, C, 1, W};
    return output_sizes;
}

at::Tensor &upsample_nearest1d_out_nocheck(at::Tensor &out, const at::Tensor &self, at::IntArrayRef output_size,
                                           c10::optional<double> scales)
{
    at::Tensor self_cp = self.unsqueeze(2);
    at_npu::native::OpCommand cmd;
    if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kHalf) {
        c10::SmallVector<int64_t, SIZE> result_size = {1, output_size[0]};
        cmd.Name("ResizeNearestNeighborV2")
            .Input(self_cp)
            .Input(result_size, at::kInt)
            .Output(out)
            .Attr("align_corners", false)
            .Attr("half_pixel_centers", false)
            .Run();
    } else {
        cmd.Name("Resize")
            .Input(self_cp)
            .Input(output_size, at::kFloat)
            .Input(output_size, at::kFloat)
            .Input(out.sizes(), at::kLong)
            .Output(out)
            .Attr("mode", static_cast<string>("nearest"))
            .Attr("nearest_mode", static_cast<string>("floor"))
            .Attr("coordinate_transformation_mode", (string) "pytorch_half_pixel")
            .Run();
    }
    out = out.squeeze(2);
    return out;
}
} // namespace

at::Tensor &upsample_nearest1d_out(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales,
                                   at::Tensor &out)
{
    c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_infer_size(self, output_size);

    npu_preparation::CheckOut({self}, out, self, op_infer_output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        upsample_nearest1d_out_nocheck(contiguous_result, self, output_size, scales);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        upsample_nearest1d_out_nocheck(out, self, output_size, scales);
    }

    return out;
}

at::Tensor upsample_nearest1d(const at::Tensor &self, at::IntArrayRef output_size, c10::optional<double> scales)
{
    c10::SmallVector<int64_t, SIZE> op_infer_output_size = upsample_nearest1d_infer_size(self, output_size);
    at::Tensor out = npu_preparation::apply_tensor(self, op_infer_output_size);

    upsample_nearest1d_out_nocheck(out, self, output_size, scales);
    return out;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest1d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    TORCH_CHECK(
        input.dim() == 3,
        "It is expected input_size equals to 3, but got size ",
        input.dim(), OPS_ERROR(ErrCode::PARAM));

    auto osize = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);
    return acl_op::upsample_nearest1d(input, osize, scales_w);
}
#endif
} // namespace acl_op
