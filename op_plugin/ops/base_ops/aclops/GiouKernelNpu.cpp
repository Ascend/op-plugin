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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, N> giou_output_size(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool is_cross) {
  c10::SmallVector<int64_t, N> output_size;
  if (is_cross) {
    output_size = {gtboxes.size(1), self.size(1)};
  } else {
    output_size = {1, self.size(1)};
  }
  return output_size;
}

at::Tensor& giou_inner_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  auto output_size = giou_output_size(self, gtboxes, is_cross);
  string mode_str = mode == 1 ? "iof" : "iou";

  at_npu::native::OpCommand cmd;
  cmd.Name("GIoU")
      .Input(self)
      .Input(gtboxes)
      .Output(result)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();
  return result;
}

std::tuple<at::Tensor&, at::Tensor&> giou_backward_inner_out_npu_nocheck(
    at::Tensor& dbboxes,
    at::Tensor& dgtboxes,
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  string mode_str = mode == 1 ? "iof" : "iou";
  at_npu::native::OpCommand cmd;
  cmd.Name("GIoUGrad")
      .Input(grad)
      .Input(bboxes)
      .Input(gtboxes)
      .Output(dbboxes)
      .Output(dgtboxes)
      .Attr("trans", trans)
      .Attr("is_cross", is_cross)
      .Attr("mode", mode_str)
      .Run();
  return std::tie(dbboxes, dgtboxes);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_giou_backward(
    const at::Tensor& grad,
    const at::Tensor& bboxes,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  TORCH_CHECK(trans && !is_cross && mode == 0,
      "giou backward only support trans==True, ",
      "is_cross==False, ",
      "mode==0('iou') current version ",
      "if you need to back propagation, ",
      "please ensure your parameter is correct!");
  // Op need form of [n] grad
  // Note: temp avoid! it'll be remove while op deal with fp16 issue!
  at::Tensor grad_cp = at::squeeze(grad, 0);
  if (grad_cp.scalar_type() == at::kHalf) {
    grad_cp = at_npu::native::custom_ops::npu_dtype_cast(grad_cp, at::kFloat);
  }
  at::Tensor bboxes_cp = bboxes;
  if (bboxes_cp.scalar_type() == at::kHalf) {
    bboxes_cp = at_npu::native::custom_ops::npu_dtype_cast(bboxes_cp, at::kFloat);
  }
  at::Tensor gtboxes_cp = gtboxes;
  if (gtboxes_cp.scalar_type() == at::kHalf) {
    gtboxes_cp = at_npu::native::custom_ops::npu_dtype_cast(gtboxes_cp, at::kFloat);
  }
  at::Tensor dbboxes = npu_preparation::ApplyTensor(bboxes_cp);
  at::Tensor dgtboxes = npu_preparation::ApplyTensor(gtboxes_cp);

  giou_backward_inner_out_npu_nocheck(dbboxes, dgtboxes, grad_cp, bboxes_cp, gtboxes_cp, trans, is_cross, mode);
  if (bboxes.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    dbboxes = at_npu::native::custom_ops::npu_dtype_cast(dbboxes, at::kHalf);
    dgtboxes = at_npu::native::custom_ops::npu_dtype_cast(dgtboxes, at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

at::Tensor npu_giou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  TORCH_CHECK(trans && !is_cross && mode == 0,
      "giou backward only support trans==True, ",
      "is_cross==False, ",
      "mode==0('iou') current version ",
      "if you need to back propagation, ",
      "please ensure your parameter is correct!");

  at::Tensor self_cp = (self.scalar_type() == at::kHalf) ?
      at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor gtboxes_cp = (gtboxes.scalar_type() == at::kHalf) ?
      at_npu::native::custom_ops::npu_dtype_cast(gtboxes, at::kFloat) : gtboxes;

  auto output_size = giou_output_size(self_cp, gtboxes_cp, is_cross);
  at::Tensor result = npu_preparation::ApplyTensor(self_cp, output_size);

  giou_inner_out_npu_nocheck(result, self_cp, gtboxes_cp, trans, is_cross, mode);
  //op's output is [1, n], same with CPU output, but pass need [n, 1].
  result = result.permute({1, 0});
  if (self.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kHalf);
  }
  return result;
}
} // namespace acl_op
