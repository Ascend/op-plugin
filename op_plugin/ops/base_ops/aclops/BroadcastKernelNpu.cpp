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
  npu_broadcast_out_nocheck(result, self, size);
  return result;
}

at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size) {
  at::Tensor self_cp = self.dtype() == at::kBool ? at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt) : self;
  at::Tensor result = npu_preparation::apply_tensor(self_cp, size);
  npu_broadcast_out_nocheck(result, self_cp, size);

  if (self.dtype() == at::kBool) {
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
  }
  return result;
}
} // namespace acl_op
