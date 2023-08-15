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
using calcu_op_util = at_npu::native::CalcuOpUtil;

at::Tensor upsample_bilinear2d(
    const at::Tensor& self_ex,
    c10::optional<at::IntArrayRef> output_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  auto osize = op_infer::upsample_infershape_with_scale(self_ex.sizes(), output_size, scale_factors);
  auto scales_h = calcu_op_util::GetScaleValue(scale_factors, 0);
  auto scales_w = calcu_op_util::GetScaleValue(scale_factors, 1);

  return op_plugin::upsample_bilinear2d(self_ex, osize, align_corners, scales_h, scales_w);
}

} // namespace op_plugin