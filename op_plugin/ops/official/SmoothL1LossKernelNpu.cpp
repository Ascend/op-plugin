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
at::Tensor& smooth_l1_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  if (self.numel() == 0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = op_plugin::npu_dtype_cast(result, at::kFloat).fill_(0);
    result = result / 0;
    return result;
  }

  string reduction_str(calcu_op_util::GetReductionStr(reduction));
  at_npu::native::OpCommand cmd;
  cmd.Name("SmoothL1LossV2")
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Attr("sigma", static_cast<float>(beta))
      .Run();
  return result;
}
} // namespace

at::Tensor& smooth_l1_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta,
    at::Tensor& result) {
  auto output_size = op_infer::smooth_l1_loss_npu_output_size(self, target, reduction);
  npu_preparation::CheckOut(
      {self, target},
      result,
      calcu_op_util::GetTensorNpuFormat(self),
      self.scalar_type(),
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    smooth_l1_loss_out_npu_nocheck(contiguous_result, self, target, reduction, beta);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    smooth_l1_loss_out_npu_nocheck(result, self, target, reduction, beta);
  }
  return result;
}

at::Tensor smooth_l1_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto output_size = op_infer::smooth_l1_loss_npu_output_size(self, target, reduction);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  smooth_l1_loss_out_npu_nocheck(result, self, target, reduction, beta);
  return result;
}
} // namespace op_plugin
