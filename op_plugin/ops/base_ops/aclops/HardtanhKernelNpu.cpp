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
at::Tensor& hardtanh_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& min,
    const at::Scalar& max) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ClipByValue")
      .Input(self)
      .Input(min, self.scalar_type())
      .Input(max, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& hardtanh_out(
    const at::Tensor& self,
    const at::Scalar& min,
    const at::Scalar& max,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    hardtanh_out_npu_nocheck(contiguous_result, self, min, max);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    hardtanh_out_npu_nocheck(result, self, min, max);
  }
    return result;
}

at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  hardtanh_out(self, min, max, result);
  return result;
}

at::Tensor& hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  return op_plugin::hardtanh_out(self, min, max, self);
}
} // namespace op_plugin
