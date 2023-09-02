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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor scatter_update(
    const at::Tensor& data,
    const at::Tensor& indices,
    const at::Tensor& updates,
    int64_t axis) {
  at::Tensor result = npu_preparation::apply_tensor(data);
  at_npu::native::OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(data)
     .Input(indices)
     .Input(updates)
     .Output(result)
     .Attr("axis", axis)
     .Run();
  return result;
}

} // namespace acl_op
