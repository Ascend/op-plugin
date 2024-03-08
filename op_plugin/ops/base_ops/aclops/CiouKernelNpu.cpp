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

#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, N> ciou_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross) {
    TORCH_CHECK(self.dim() == 2, "ciou expected input in 2D, "
        "but input self has sizes ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gtboxes.dim() == 2, "ciou expected input in 2D, "
        "but input gtboxes has sizes ", gtboxes.dim(),
        OPS_ERROR(ErrCode::PARAM));
  c10::SmallVector<int64_t, N> output_size;
  if (is_cross) {
    output_size = {gtboxes.size(1), self.size(1)};
  } else {
    output_size = {1, self.size(1)};
  }
  return output_size;
}

std::tuple<at::Tensor, at::Tensor> ciou_inner_out_npu(
    at::Tensor& overlap,
    at::Tensor& atan_sub,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
  string mode_str = mode == 1 ? "iof" : "iou";
  at_npu::native::OpCommand cmd;
  cmd.Name("CIoU")
      .Input(self)
      .Input(gtboxes)
      .Output(overlap)
      .Output(atan_sub)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Attr("atan_sub_flag", atan_sub_flag)
      .Run();
  return std::tie(overlap, atan_sub);
}

std::tuple<at::Tensor&, at::Tensor&> ciou_backward_inner_out_npu(
    at::Tensor& dbboxes,
    at::Tensor& dgtboxes,
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const at::Tensor& atan_sub,
    bool trans,
    bool is_cross,
    int64_t mode) {
  string mode_str = mode == 1 ? "iof" : "iou";
  at_npu::native::OpCommand cmd;
  cmd.Name("CIoUGrad")
      .Input(grad)
      .Input(bboxes)
      .Input(gtboxes)
      .Input(atan_sub)
      .Output(dbboxes)
      .Output(dgtboxes)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();

  return std::tie(dbboxes, dgtboxes);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
  bool self_is_half = self.scalar_type() == at::kHalf;
  bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
  at::Tensor self_cp = self_is_half ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

  auto output_size = ciou_output_size(self_cp, gtboxes_cp, is_cross);
  at::Tensor overlap = npu_preparation::apply_tensor(self_cp, output_size);
  at::Tensor atan_sub = npu_preparation::apply_tensor(self_cp, output_size);
  ciou_inner_out_npu(overlap, atan_sub, self_cp, gtboxes_cp, trans, is_cross, mode, atan_sub_flag);
  if (self_is_half || gtboxes_is_half) {
    overlap = at_npu::native::custom_ops::npu_dtype_cast(overlap, at::kHalf);
  }
  return std::tie(overlap, atan_sub);
}

std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    const c10::optional<at::Tensor>& atan_sub_opt,
    bool trans,
    bool is_cross,
    int64_t mode) {
  const at::Tensor& atan_sub = c10::value_or_else(atan_sub_opt, [] {return at::Tensor();});
  at::Tensor grad_cp = at::squeeze(grad, 0);
  if (grad_cp.scalar_type() == at::kHalf) {
    grad_cp = at_npu::native::custom_ops::npu_dtype_cast(grad_cp, at::kFloat);
  }
  bool bboxes_is_half = bboxes.scalar_type() == at::kHalf;
  bool gtboxes_is_half = gtboxes.scalar_type() == at::kHalf;
  at::Tensor bboxes_cp = bboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(bboxes, at::kFloat) : bboxes;
  at::Tensor gtboxes_cp = gtboxes_is_half ? at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

  at::Tensor dbboxes = npu_preparation::apply_tensor(bboxes_cp);
  at::Tensor dgtboxes = npu_preparation::apply_tensor(gtboxes_cp);
  ciou_backward_inner_out_npu(dbboxes, dgtboxes, grad_cp, bboxes_cp, gtboxes_cp, atan_sub, trans, is_cross, mode);
  if (bboxes_is_half || gtboxes_is_half) {
    dbboxes = at_npu::native::custom_ops::npu_dtype_cast(dbboxes, at::kHalf);
    dgtboxes = at_npu::native::custom_ops::npu_dtype_cast(dgtboxes, at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

at::Tensor npu_ciou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode,
    bool atan_sub_flag) {
  auto results = at_npu::native::custom_ops::_npu_ciou(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
  return std::get<0>(results);
}
} // namespace acl_op
