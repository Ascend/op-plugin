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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& mse_loss_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction) {
  if (self.numel() == 0 || target.numel() == 0) {
    // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
    result = op_plugin::npu_dtype_cast(result, at::kFloat);
    op_plugin::fill_(result, 0);
    result = result / 0;
    return result;
  }
  auto unified_result = npu_preparation::binary_op_check(result, self, target, true);
  string reduction_str(op_plugin::utils::get_reduction_str(reduction));
  at_npu::native::OpCommand cmd;
  cmd.Name("MseLoss")
      .Expect(unified_result)
      .Input(self)
      .Input(target)
      .Output(result)
      .Attr("reduction", reduction_str)
      .Run();
  return result;
}
} // namespace

at::Tensor& mse_loss_out(const at::Tensor& self, const at::Tensor& target, int64_t reduction, at::Tensor& result) {
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::input_same_output_size(self);
  }

  npu_preparation::CheckOut(
      {self, target},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    mse_loss_out_npu_nocheck(contiguous_result, self, target, reduction);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    mse_loss_out_npu_nocheck(result, self, target, reduction);
  }
  return result;
}

at::Tensor mse_loss(const at::Tensor& self, const at::Tensor& target, int64_t reduction) {
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::input_same_output_size(self);
  }
  at::Tensor result = (reduction == at::Reduction::None) ?
      npu_preparation::ApplyTensor(self, output_size) :
      npu_preparation::ApplyTensorWithFormat(self, output_size, ACL_FORMAT_ND);

  mse_loss_out_npu_nocheck(result, self, target, reduction);
  return result;
}
} // namespace op_plugin
