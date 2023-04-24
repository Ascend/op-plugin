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

at::Tensor& reciprocal_out_npu_nocheck(at::Tensor& result, const at::Tensor& self) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Reciprocal")
      .Input(self)
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor& reciprocal_out(const at::Tensor& self, at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);
  npu_preparation::CheckMemory({self}, {result});
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    reciprocal_out_npu_nocheck(contiguous_result, self);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    reciprocal_out_npu_nocheck(result, self);
  }
  return result;
}

at::Tensor reciprocal(const at::Tensor& self) {
  at::Tensor self_cp = isIntegralType(self.scalar_type(), true) ?
      op_plugin::npu_dtype_cast(self, at::kFloat) : self;
  at::Tensor result = npu_preparation::ApplyTensor(self_cp);
  reciprocal_out_npu_nocheck(result, self_cp);

  return result;
}

at::Tensor& reciprocal_(at::Tensor& self) {
  op_plugin::reciprocal_out(self, self);

  return self;
}

} // namespace op_plugin