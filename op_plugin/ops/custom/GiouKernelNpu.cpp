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

#include <torch/csrc/autograd/custom_function.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
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

at::Tensor giou_npu(
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

  at::Tensor self_cp = self;
  if (self_cp.scalar_type() == at::kHalf) {
    self_cp = op_plugin::npu_dtype_cast(self_cp, at::kFloat);
  }
  at::Tensor gtboxes_cp = gtboxes;
  if (gtboxes_cp.scalar_type() == at::kHalf) {
    gtboxes_cp = op_plugin::npu_dtype_cast(gtboxes_cp, at::kFloat);
  }
  auto output_size = giou_output_size(self_cp, gtboxes_cp, is_cross);
  at::Tensor result = npu_preparation::ApplyTensor(self_cp, output_size);

  giou_inner_out_npu_nocheck(result, self_cp, gtboxes_cp, trans, is_cross, mode);
  //op's output is [1, n], same with CPU output, but pass need [n, 1].
  result = result.permute({1, 0});
  if (self.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    result = op_plugin::npu_dtype_cast(result, at::kHalf);
  }
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
    grad_cp = op_plugin::npu_dtype_cast(grad_cp, at::kFloat);
  }
  at::Tensor bboxes_cp = bboxes;
  if (bboxes_cp.scalar_type() == at::kHalf) {
    bboxes_cp = op_plugin::npu_dtype_cast(bboxes_cp, at::kFloat);
  }
  at::Tensor gtboxes_cp = gtboxes;
  if (gtboxes_cp.scalar_type() == at::kHalf) {
    gtboxes_cp = op_plugin::npu_dtype_cast(gtboxes_cp, at::kFloat);
  }
  at::Tensor dbboxes = npu_preparation::ApplyTensor(bboxes_cp);
  at::Tensor dgtboxes = npu_preparation::ApplyTensor(gtboxes_cp);

  giou_backward_inner_out_npu_nocheck(dbboxes, dgtboxes, grad_cp, bboxes_cp, gtboxes_cp, trans, is_cross, mode);
  if (bboxes.scalar_type() == at::kHalf || gtboxes.scalar_type() == at::kHalf) {
    dbboxes = op_plugin::npu_dtype_cast(dbboxes, at::kHalf);
    dgtboxes = op_plugin::npu_dtype_cast(dgtboxes, at::kHalf);
  }
  return std::tie(dbboxes, dgtboxes);
}

class NPUGiouFunction : public torch::autograd::Function<NPUGiouFunction> {
public:
  static at::Tensor forward(
      AutogradContext *ctx,
      const at::Tensor& self,
      const at::Tensor& gtboxes,
      bool trans,
      bool is_cross,
      int64_t mode) {
    ctx->saved_data["trans"] = trans;
    ctx->saved_data["is_cross"] = is_cross;
    ctx->saved_data["mode"] = mode;
    at::AutoNonVariableTypeMode g;
    ctx->save_for_backward({self, gtboxes});
    return giou_npu(self, gtboxes, trans, is_cross, mode);
  }

  static tensor_list backward(
      AutogradContext *ctx,
      tensor_list grad_outputs) {
    auto trans = ctx->saved_data["trans"].toBool();
    auto is_cross = ctx->saved_data["is_cross"].toBool();
    auto mode = ctx->saved_data["mode"].toInt();
    auto saved = ctx->get_saved_variables();
    auto bboxes = saved[0];
    auto gtboxes = saved[1];

    std::tuple<at::Tensor, at::Tensor> result =
        op_plugin::npu_giou_backward(grad_outputs[0], bboxes, gtboxes, trans, is_cross, mode);

    tensor_list output = {std::get<0>(result),
        std::get<1>(result),
        at::Tensor(),
        at::Tensor(),
        at::Tensor()};
    return output;
  }
};

at::Tensor npu_giou(
    const at::Tensor& self,
    const at::Tensor& gtboxes,
    bool trans,
    bool is_cross,
    int64_t mode) {
  return NPUGiouFunction::apply(self, gtboxes, trans, is_cross, mode);
}
} // namespace op_plugin
