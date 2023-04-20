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

namespace{
at::Tensor& npu_broadcast_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef size) {
  at_npu::native::OpCommand cmd;
  cmd.Name("BroadcastTo")
      .Input(self)
      .Input(size)
      .Output(result)
      .Run();
  return result;
}
} // namespace

at::Tensor& npu_broadcast_out(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self},
      result,
      self,
      size);
  bool self_is_bool = self.dtype() == at::kBool;
  at::Tensor self_cp = self_is_bool ? op_plugin::npu_dtype_cast(self, at::kInt) : self;
  at::Tensor result_cp = self_is_bool ? op_plugin::npu_dtype_cast(result, at::kInt) : result;

  if (!npu_utils::check_match(&result_cp)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
    npu_broadcast_out_nocheck(contiguous_result, self_cp, size);
    npu_utils::format_fresh_view(result_cp, contiguous_result);
  } else {
    npu_broadcast_out_nocheck(result_cp, self_cp, size);
  }
  if (self_is_bool) {
    result_cp = op_plugin::npu_dtype_cast(result_cp, at::kBool);
    result.copy_(result_cp);
  } else {
    result = result_cp;
  }
  return result;
}

at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size) {
  at::Tensor self_cp = self.dtype() == at::kBool ? op_plugin::npu_dtype_cast(self, at::kInt) : self;
  at::Tensor result = npu_preparation::ApplyTensor(self_cp, size);
  npu_broadcast_out_nocheck(result, self_cp, size);

  if (self.dtype() == at::kBool) {
    result = op_plugin::npu_dtype_cast(result, at::kBool);
  }
  return result;
}
} // namespace op_plugin
