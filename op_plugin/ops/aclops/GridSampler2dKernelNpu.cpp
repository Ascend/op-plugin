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

at::Tensor grid_sampler_2d(
    const at::Tensor& self,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners)
{
    TORCH_CHECK((0 <= interpolation_mode && interpolation_mode <= 2), "interpolation_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK((0 <= padding_mode && padding_mode <= 2), "padding_mode must be in range [0~2]."
        + OPS_ERROR(ErrCode::VALUE));

    at::Tensor dtype_cast_of_self = self;
    at::Tensor dtype_cast_of_grid = grid;
    if (dtype_cast_of_self.scalar_type() == c10::ScalarType::Half) {
        dtype_cast_of_self = at_npu::native::custom_ops::npu_dtype_cast(dtype_cast_of_self, c10::ScalarType::Float);
    }
    if (dtype_cast_of_grid.scalar_type() == c10::ScalarType::Half) {
        dtype_cast_of_grid = at_npu::native::custom_ops::npu_dtype_cast(dtype_cast_of_grid, c10::ScalarType::Float);
    }

    c10::SmallVector<int64_t, SIZE> output_size = {
        dtype_cast_of_self.size(0),
        dtype_cast_of_self.size(1),
        dtype_cast_of_grid.size(1),
        dtype_cast_of_grid.size(2)};

    at::Tensor result = npu_preparation::apply_tensor_with_format(dtype_cast_of_self, output_size, ACL_FORMAT_ND);
    std::string inter_mode[] = {"bilinear", "nearest", "bicubic"};
    std::string pad_mode[] = {"zeros", "border", "reflection"};
    at_npu::native::OpCommand cmd;
    cmd.Name("GridSampler2D")
        .Input(dtype_cast_of_self)
        .Input(dtype_cast_of_grid)
        .Output(result)
        .Attr("interpolation_mode", inter_mode[interpolation_mode])
        .Attr("padding_mode", pad_mode[padding_mode])
        .Attr("align_corners", align_corners)
        .Run();

    c10::ScalarType self_scalar_type(self.scalar_type());
    if (result.scalar_type() != self_scalar_type) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, self_scalar_type);
    }
    return result;
}
}  // op_plugin
