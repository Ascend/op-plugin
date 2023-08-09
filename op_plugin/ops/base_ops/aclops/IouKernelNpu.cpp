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
using calcu_op_util = at_npu::native::CalcuOpUtil;

at::Tensor npu_iou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode) {
  at::Tensor bboxes_fp16 = bboxes;
  if (bboxes.scalar_type() != at::ScalarType::Half) {
    bboxes_fp16 = op_plugin::npu_dtype_cast(bboxes, at::kHalf);
  }
  at::Tensor gtboxes_fp16 = gtboxes;
  if (gtboxes.scalar_type() != at::ScalarType::Half) {
    gtboxes_fp16 = op_plugin::npu_dtype_cast(gtboxes, at::kHalf);
  }

  auto output_size = {gtboxes.size(0), bboxes.size(0)};
  at::Tensor overlap = npu_preparation::ApplyTensorWithFormat(
      bboxes_fp16,
      output_size,
      calcu_op_util::GetTensorNpuFormat(bboxes));
  string mode_str = "iou";
  if (mode == 1) {
    mode_str = "iof";
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("Iou")
      .Input(bboxes_fp16)
      .Input(gtboxes_fp16)
      .Output(overlap)
      .Attr("mode", mode_str)
      .Attr("eps", static_cast<float>(0.01))
      .Run();

  if (overlap.scalar_type() != bboxes.scalar_type()) {
    overlap = op_plugin::npu_dtype_cast(overlap, bboxes.scalar_type());
  }
  return overlap;
}

at::Tensor npu_ptiou(
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    int64_t mode) {
  return op_plugin::npu_iou(bboxes, gtboxes, mode);
}
} // namespace op_plugin
