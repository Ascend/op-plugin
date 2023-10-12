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
at::Tensor &upsample_nearest2d_backward_out_nocheck(at::Tensor &y, const at::Tensor &grads, at::IntArrayRef output_size,
                                                    at::IntArrayRef input_size, c10::optional<double> scales_h,
                                                    c10::optional<double> scales_w)
{
    TORCH_CHECK(input_size.size() == 4, "The length of input_size should be equal to 4, but got ", input_size.size());

    at::SmallVector<int64_t, N> output_sizes = {input_size[2], input_size[3]};
    at_npu::native::OpCommand cmd;
    cmd.Name("ResizeNearestNeighborV2Grad")
        .Input(grads, "grads")
        .Input(output_sizes, at::kInt)
        .Output(y, "y")
        .Attr("align_corners", false)
        .Attr("half_pixel_centers", false)
        .Run();

    return y;
}
} // namespace

at::Tensor &upsample_nearest2d_backward_out(const at::Tensor &grads, at::IntArrayRef output_size,
                                            at::IntArrayRef input_size, c10::optional<double> scales_h,
                                            c10::optional<double> scales_w, at::Tensor &y)
{
    npu_preparation::CheckOut({grads}, y, npu_preparation::get_tensor_npu_format(y), grads.scalar_type(), input_size);

    if (!npu_utils::check_match(&y)) {
        at::Tensor contiguous_y = npu_utils::format_contiguous(y);
        upsample_nearest2d_backward_out_nocheck(contiguous_y, grads, output_size, input_size, scales_h, scales_w);
        npu_utils::format_fresh_view(y, contiguous_y);
    } else {
        upsample_nearest2d_backward_out_nocheck(y, grads, output_size, input_size, scales_h, scales_w);
    }

    return y;
}

at::Tensor upsample_nearest2d_backward(const at::Tensor &grad_output, at::IntArrayRef output_size,
                                       at::IntArrayRef input_size, c10::optional<double> scales_h,
                                       c10::optional<double> scales_w)
{
    at::Tensor grads = grad_output;
    if (grad_output.scalar_type() != at::ScalarType::Float) {
        grads = at_npu::native::custom_ops::npu_dtype_cast(grad_output, at::kFloat);
    }
    at::Tensor grad_input = npu_preparation::apply_tensor(input_size, grads.options(), grad_output);
    upsample_nearest2d_backward_out_nocheck(grad_input, grads, output_size, input_size, scales_h, scales_w);
    return grad_input;
}
} // namespace acl_op
