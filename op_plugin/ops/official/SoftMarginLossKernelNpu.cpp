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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& soft_margin_loss_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  at::Tensor target_broadcast = target;
  if(target.sizes() != self.sizes()) {
    target_broadcast = op_plugin::npu_broadcast(target, self.sizes());
  }
  string reduction_str(calcu_op_util::GetReductionStr(reduction));
  at_npu::native::OpCommand cmd;
  cmd.Name("SoftMarginLoss")
      .Input(self)
      .Input(target_broadcast)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Run();
  return result;
}
} // namespace

at::Tensor& soft_margin_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& result) {
  auto output_size = op_infer::soft_margin_loss_npu_output_size(self, target, reduction);
  npu_preparation::CheckOut(
      {self, target},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    soft_margin_loss_out_nocheck(contiguous_result, self, target, reduction);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    soft_margin_loss_out_nocheck(result, self, target, reduction);
  }
  return result;
}

at::Tensor soft_margin_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
  auto output_size = op_infer::soft_margin_loss_npu_output_size(self, target, reduction);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);

  soft_margin_loss_out_nocheck(result, self, target, reduction);

  if (reduction == at::Reduction::None) {
    return result;
  } else {
    return result.reshape({});
  }
}
} // namespace op_plugin