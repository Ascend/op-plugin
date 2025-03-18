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

std::tuple<at::Tensor, at::Tensor> grid_sampler_2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask)
{
    TORCH_CHECK((0 <= interpolation_mode && interpolation_mode <= 2), "interpolation_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK((0 <= padding_mode && padding_mode <= 2), "padding_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));

    at::Tensor format_cast_of_grad = grad_output;
    at::Tensor format_cast_of_input = input;
    at::Tensor format_cast_of_grid = grid;

    if (format_cast_of_grad.scalar_type() == at::ScalarType::Half) {
        format_cast_of_grad = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_grad, at::ScalarType::Float);
    }
    if (format_cast_of_input.scalar_type() == at::ScalarType::Half) {
        format_cast_of_input = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_input, at::ScalarType::Float);
    }
    if (format_cast_of_grid.scalar_type() == at::ScalarType::Half) {
        format_cast_of_grid = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_grid, at::ScalarType::Float);
    }

    at::Tensor dx = npu_preparation::apply_tensor(format_cast_of_input);
    at::Tensor dgrid = npu_preparation::apply_tensor(format_cast_of_grid);

    c10::SmallVector<string, SIZE> inter_mode = {"bilinear", "nearest", "bicubic"};
    c10::SmallVector<string, SIZE> pad_mode = {"zeros", "border", "reflection"};

    at_npu::native::OpCommand cmd;
    cmd.Name("GridSampler2DGrad")
        .Input(format_cast_of_grad)
        .Input(format_cast_of_input)
        .Input(format_cast_of_grid)
        .Output(dx)
        .Output(dgrid)
        .Attr("interpolation_mode", inter_mode[interpolation_mode])
        .Attr("padding_mode", pad_mode[padding_mode])
        .Attr("align_corners", align_corners)
        .Run();

    at::ScalarType input_scalar_type(input.scalar_type());
    if (dx.scalar_type() != input_scalar_type) {
        dx = at_npu::native::custom_ops::npu_dtype_cast(dx, input_scalar_type);
        dgrid = at_npu::native::custom_ops::npu_dtype_cast(dgrid, input_scalar_type);
    }

    return std::tuple<at::Tensor, at::Tensor>(dx, dgrid);
}
} // namespace acl_op
