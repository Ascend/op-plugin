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
at::Tensor& nonzero_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  c10::SmallVector<int64_t, N> output_sync_idx = {0};
  at_npu::native::OpCommand cmd;
  cmd.Sync(output_sync_idx)
      .Name("NonZero")
      .Input(self)
      .Output(result)
      .Attr("transpose", false)
      .Run();
  return result;
}
} // namespace

at::Tensor& nonzero_out(const at::Tensor& self, at::Tensor& result) {
  auto output_size = op_infer::nonzero_npu_max_output_size(self);
  npu_preparation::CheckOut(
      {self},
      result,
      calcu_op_util::GetTensorNpuFormat(self),
      at::ScalarType::Long,
      output_size);

  nonzero_out_npu_nocheck(result, self);
  return result;
}

at::Tensor nonzero(const at::Tensor& self) {
  auto output_size = op_infer::nonzero_npu_max_output_size(self);
  at::Tensor result = npu_preparation::apply_tensor(output_size, self.options().dtype(at::kLong), self);
  nonzero_out_npu_nocheck(result, self);
  return result;
}
} // namespace op_plugin