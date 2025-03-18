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

namespace {
at::Tensor &grid_sampler_3d_npu_nocheck(at::Tensor &result, const at::Tensor &input, const at::Tensor &grid,
                                        std::string inter_mode, std::string padding_mode, bool align_corners)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GridSampler3D")
        .Input(input)
        .Input(grid)
        .Output(result)
        .Attr("interpolation_mode", inter_mode)
        .Attr("padding_mode", padding_mode)
        .Attr("align_corners", align_corners)
        .Run();
    return result;
}
} // namespace

at::Tensor grid_sampler_3d(const at::Tensor &input, const at::Tensor &grid, int64_t interpolation_mode,
                           int64_t padding_mode, bool align_corners)
{
    TORCH_CHECK((0 <= interpolation_mode && interpolation_mode <= 2), "interpolation_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK((0 <= padding_mode && padding_mode <= 2), "padding_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(input.dim() == 5 && input.dim() == grid.dim(),
        "grid_sampler(): expected 5D input and grid with same number of "
        "dimensions, but got input with sizes ",
        input.sizes(), " and grid with sizes ", grid.sizes(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.defined(), "grid_sampler(): expected input to not be undefined"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grid.defined(), "grid_sampler(): expected grid to not be undefined"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input.size(0) == grid.size(0),
        "grid_sampler(): expected grid and input to have same batch "
        "size, but got "
        "input with sizes ",
        input.sizes(), " and grid with sizes ", grid.sizes(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grid.size(-1) == input.dim() - 2, "grid_sampler(): expected grid to have size ", input.dim() - 2,
        " in last "
        "dimension, but got grid with sizes ",
        grid.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    at::Tensor format_cast_of_self = input;
    at::Tensor format_cast_of_grid = grid;
    if (format_cast_of_self.scalar_type() == at::ScalarType::Half) {
        format_cast_of_self = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_self, at::ScalarType::Float);
    }
    if (format_cast_of_grid.scalar_type() == at::ScalarType::Half) {
        format_cast_of_grid = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_grid, at::ScalarType::Float);
    }

    c10::SmallVector<int64_t, SIZE> output_size = {format_cast_of_self.size(0), format_cast_of_self.size(1),
                                                   format_cast_of_grid.size(1), format_cast_of_grid.size(2),
                                                   format_cast_of_grid.size(3)};

    at::Tensor result =
        npu_preparation::apply_tensor_with_format(output_size, format_cast_of_self.options(), ACL_FORMAT_ND);
    std::string inter_mode[] = {"bilinear", "nearest", "bicubic"};
    std::string pad_mode[] = {"zeros", "border", "reflection"};

    grid_sampler_3d_npu_nocheck(result, format_cast_of_self, format_cast_of_grid, inter_mode[interpolation_mode],
                                pad_mode[padding_mode], align_corners);

    at::ScalarType self_scalar_type(input.scalar_type());
    if (result.scalar_type() != self_scalar_type) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, self_scalar_type);
    }
    return result;
}
} // namespace acl_op
