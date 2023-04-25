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
at::Tensor& logaddexp2_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other) {
  at::Tensor self_exp2 = npu_preparation::ApplyTensor(self);
  at::Tensor other_exp2 = npu_preparation::ApplyTensor(self);

  at::Scalar base(2);

  at_npu::native::OpCommand cmd_exp2_1, cmd_exp2_2, cmd_add, cmd_log;
  cmd_exp2_1.Name("Pow")
      .Input(base, self.scalar_type())
      .Input(self)
      .Output(self_exp2)
      .Run();

  cmd_exp2_2.Name("Pow")
      .Input(base, other.scalar_type())
      .Input(other)
      .Output(other_exp2)
      .Run();

  at::Tensor add_result = npu_preparation::ApplyTensor(self);
  auto unified_result = npu_preparation::binary_op_check(add_result, self_exp2, other_exp2, true);

  cmd_add.Name("Add")
      .Expect(unified_result)
      .Input(self_exp2)
      .Input(other_exp2)
      .Output(add_result)
      .Run();

  cmd_log.Name("Log")
      .Input(add_result)
      .Output(result)
      .Attr("base", (float)2.0)
      .Attr("scale", (float)1.0)
      .Attr("shift", (float)0.0)
      .Run();

  return result;
}
} // namespace

at::Tensor& logaddexp2_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    logaddexp2_out_npu_nocheck(contiguous_result, self, other);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    logaddexp2_out_npu_nocheck(result, self, other);
  }
  return result;
}

at::Tensor logaddexp2(const at::Tensor& self, const at::Tensor& other) {
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  logaddexp2_out_npu_nocheck(result, self, other);
  return result;
}
} // namespace op_plugin
