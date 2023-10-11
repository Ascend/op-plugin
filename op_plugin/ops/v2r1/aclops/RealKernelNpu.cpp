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
using npu_utils = at_npu::native::NpuUtils;

at::Tensor real(const at::Tensor& self)
{
  TORCH_NPU_WARN_ONCE("Warning: kernel [real] is not supported by NPU currently. Now this kernel is running on CPU.");
  at::Tensor self_cpu = self.to("cpu");
  auto result = at::native::real(self_cpu);
  at::Tensor output = result.to(self.device());
  return output;
}
}  // namespace acl_op