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
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&> multilabel_margin_loss_forward_out_nocheck(
    at::Tensor& output,
    at::Tensor& is_target,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  string reduction_str = op_plugin::utils::get_reduction_str(reduction);
  at_npu::native::OpCommand cmd;
  cmd.Name("MultilabelMarginLoss")
      .Input(self)
      .Input(target)
      .Output(output)
      .Output(is_target)
      .Attr("reduction", reduction_str)
      .Run();
  return std::tuple<at::Tensor&, at::Tensor&>(output, is_target);
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> multilabel_margin_loss_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& output,
    at::Tensor& is_target) {
  c10::SmallVector<int64_t, SIZE> output_size;
  int64_t nframe;
  if (self.dim() <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }
  if (reduction == at::Reduction::None) {
    output_size = {nframe};
  }
  npu_preparation::CheckOut(
      {self, target},
      output,
      self,
      output_size);
  npu_preparation::CheckOut(
      {self, target},
      is_target,
      target);

  bool output_match = npu_utils::check_match(&output);
  bool is_target_match = npu_utils::check_match(&is_target);
  if (!(output_match && is_target_match)) {
    at::Tensor contiguous_output = output_match ? output : npu_utils::format_contiguous(output);
    at::Tensor contiguous_is_target = is_target_match ? is_target : npu_utils::format_contiguous(is_target);
    multilabel_margin_loss_forward_out_nocheck(contiguous_output, contiguous_is_target, self, target, reduction);
    if (!output_match) {
      npu_utils::format_fresh_view(output, contiguous_output);
    }
    if (!is_target_match) {
      npu_utils::format_fresh_view(is_target, contiguous_is_target);
    }
  } else {
    multilabel_margin_loss_forward_out_nocheck(output, is_target, self, target, reduction);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(output, is_target);
}

std::tuple<at::Tensor, at::Tensor> multilabel_margin_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  c10::SmallVector<int64_t, SIZE> output_size;
  int64_t nframe;
  if (self.dim() <= 1) {
    nframe = 1;
  } else {
    nframe = self.size(0);
  }
  if (reduction == at::Reduction::None) {
    output_size = {nframe};
  }
  auto output = npu_preparation::apply_tensor(self, output_size);
  auto is_target = npu_preparation::apply_tensor(target);

  acl_op::multilabel_margin_loss_forward_out(
      self,
      target,
      reduction,
      output,
      is_target);
  return std::make_tuple(output, is_target);
}
} // namespace acl_op
