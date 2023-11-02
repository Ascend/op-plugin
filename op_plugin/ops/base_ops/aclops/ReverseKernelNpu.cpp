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

namespace {
at::Tensor& reverse_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef axis) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ReverseV2")
      .Input(self)
      .Input(axis, at::kInt)
      .Output(result)
      .Run();

  return result;
}
} // namespace

at::Tensor reverse(
    const at::Tensor& self,
    at::IntArrayRef axis) {
  at::Tensor result = npu_preparation::apply_tensor(self);
  reverse_out_nocheck(result, self, axis);

  return result;
}
} // namespace acl_op
