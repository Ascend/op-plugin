// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& grid_sampler_3d_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& grid,
    std::string inter_mode,
    std::string padding_mode,
    bool align_corners) {
  at_npu::native::OpCommand cmd;
  cmd.Name("GridSampler3D")
      .Input(self)
      .Input(grid)
      .Output(result)
      .Attr("interpolation_mode", inter_mode)
      .Attr("padding_mode", padding_mode)
      .Attr("align_corners", align_corners)
      .Run();
  return result;
}
} // namespace

at::Tensor grid_sampler_3d(
    const at::Tensor& self,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  TORCH_CHECK((0 <= interpolation_mode && interpolation_mode <= 2), "interpolation_mode must be in range [0~2].")
  TORCH_CHECK((0 <= padding_mode && padding_mode <= 2), "padding_mode must be in range [0~2].")
  at::Tensor format_cast_of_self = self;
  at::Tensor format_cast_of_grid = grid;
  if (format_cast_of_self.scalar_type() == at::ScalarType::Half) {
    format_cast_of_self = op_plugin::npu_dtype_cast(format_cast_of_self, at::ScalarType::Float);
  }
  if (format_cast_of_grid.scalar_type() == at::ScalarType::Half) {
    format_cast_of_grid = op_plugin::npu_dtype_cast(format_cast_of_grid, at::ScalarType::Float);
  }

  c10::SmallVector<int64_t, SIZE> output_size = {
      format_cast_of_self.size(0),
      format_cast_of_self.size(1),
      format_cast_of_grid.size(1),
      format_cast_of_grid.size(2),
      format_cast_of_grid.size(3)};

  at::Tensor result =
      npu_preparation::ApplyTensorWithFormat(output_size, format_cast_of_self.options(), ACL_FORMAT_ND);
  std::string inter_mode[] = {"bilinear", "nearest", "bicubic"};
  std::string pad_mode[] = {"zeros", "border", "reflection"};

  grid_sampler_3d_npu_nocheck(
      result,
      format_cast_of_self,
      format_cast_of_grid,
      inter_mode[interpolation_mode],
      pad_mode[padding_mode],
      align_corners);

  at::ScalarType self_scalar_type(self.scalar_type());
  if (result.scalar_type() != self_scalar_type) {
    result = op_plugin::npu_dtype_cast(result, self_scalar_type);
  }
  return result;
}
} // namespace op_plugin
