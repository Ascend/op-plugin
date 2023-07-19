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
std::tuple<at::Tensor&, at::Tensor&> grid_sampler_3d_backward_nocheck(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    std::string inter_mode,
    std::string padding_mode,
    bool align_corners,
    at::Tensor& dx,
    at::Tensor& dgrid) {
  at_npu::native::OpCommand cmd;
  cmd.Name("GridSampler3DGrad")
      .Input(grad)
      .Input(input)
      .Input(grid)
      .Output(dx)
      .Output(dgrid)
      .Attr("interpolation_mode", inter_mode)
      .Attr("padding_mode", padding_mode)
      .Attr("align_corners", align_corners)
      .Run();
  return std::tie(dx, dgrid);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> grid_sampler_3d_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool,2> output_mask) {
  TORCH_CHECK(
      (0 <= interpolation_mode && interpolation_mode <= 2),
      "interpolation_mode must be in range [0~2].")
  TORCH_CHECK(
      (0 <= padding_mode && padding_mode <= 2),
      "padding_mode must be in range [0~2].")
  at::Tensor format_cast_of_grad = grad;
  at::Tensor format_cast_of_input = input;
  at::Tensor format_cast_of_grid = grid;
  if (format_cast_of_grad.scalar_type() == at::ScalarType::Half) {
    format_cast_of_grad = op_plugin::npu_dtype_cast(format_cast_of_grad, at::ScalarType::Float);
  }
  if (format_cast_of_input.scalar_type() == at::ScalarType::Half) {
    format_cast_of_input = op_plugin::npu_dtype_cast(format_cast_of_input, at::ScalarType::Float);
  }
  if (format_cast_of_grid.scalar_type() == at::ScalarType::Half) {
    format_cast_of_grid = op_plugin::npu_dtype_cast(format_cast_of_grid, at::ScalarType::Float);
  }

  at::Tensor dx = npu_preparation::apply_tensor(format_cast_of_input);
  at::Tensor dgrid = npu_preparation::apply_tensor(format_cast_of_grid);
  std::string inter_mode_list[] = {"bilinear", "nearest", "bicubic"};
  std::string padding_mode_list[] = {"zeros", "border", "reflection"};

  grid_sampler_3d_backward_nocheck(format_cast_of_grad, format_cast_of_input, format_cast_of_grid,
      inter_mode_list[interpolation_mode], padding_mode_list[padding_mode], align_corners, dx, dgrid);
  return std::tie(dx, dgrid);
}
} // namespace op_plugin