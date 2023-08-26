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
c10::SmallVector<int64_t, SIZE> ger_npu_output_size(const at::Tensor& self, const at::Tensor& vec2) {
  int64_t output_size_0 = self.size(0);
  int64_t output_size_1 = vec2.size(0);
  c10::SmallVector<int64_t, SIZE> output_size = {output_size_0, output_size_1};

  return output_size;
}

at::Tensor& ger_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& vec2) {
  at_npu::native::OpCommand cmd;
  cmd.Name("Ger")
      .Input(self)
      .Input(vec2)
      .Output(result)
      .Run();
  return result;
}
}  // namespace

at::Tensor& ger_out(const at::Tensor& self, const at::Tensor& vec2, at::Tensor& result) {
  TORCH_CHECK(self.dim() == 1, "Input1 must have only 1 dims.");
  TORCH_CHECK(vec2.dim() == 1, "Input2 must have only 1 dims.");

  auto output_size = ger_npu_output_size(self, vec2);
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    ger_out_npu_nocheck(contiguous_result, self, vec2);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    ger_out_npu_nocheck(result, self, vec2);
  }

  return result;
}

at::Tensor ger(const at::Tensor& self, const at::Tensor& vec2) {
  TORCH_CHECK(self.dim() == 1, "Input1 must have only 1 dims.");
  TORCH_CHECK(vec2.dim() == 1, "Input2 must have only 1 dims.");

  auto output_size = ger_npu_output_size(self, vec2);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  ger_out_npu_nocheck(result, self, vec2);

  return result;
}
}  // namespace acl_op
