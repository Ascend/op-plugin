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
at::Tensor& logical_not_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  at::ScalarType src_type = self.scalar_type();
  at::Tensor self_cast = self;

  if (src_type != at::ScalarType::Bool) {
    self_cast = op_plugin::npu_dtype_cast(self, at::kBool);
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }

  at_npu::native::OpCommand cmd;
  cmd.Name("LogicalNot")
      .Input(self_cast)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& logical_not_out(const at::Tensor& self, at::Tensor& result) {
  auto result_dtype = result.scalar_type();
  npu_preparation::CheckOut(
      {self},
      result,
      calcu_op_util::GetTensorNpuFormat(self),
      result_dtype,
      self.sizes());

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    logical_not_out_npu_nocheck(contiguous_result, self);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    logical_not_out_npu_nocheck(result, self);
  }

  if (self.scalar_type() != at::ScalarType::Bool) {
    result = op_plugin::npu_dtype_cast(result, result_dtype);
  }
  return result;
}

at::Tensor logical_not(const at::Tensor& self) {
  at::Tensor result = npu_preparation::ApplyTensor(
      self.sizes(),
      self.options().dtype(at::kBool),
      self);
  logical_not_out_npu_nocheck(result, self);
  return result;
}

at::Tensor& logical_not_(at::Tensor& self) {
  return op_plugin::logical_not_out(self, self);
}
} // namespace op_plugin
