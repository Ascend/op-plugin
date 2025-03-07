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

at::Tensor& sinc_out(const at::Tensor& self, at::Tensor& result)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [sinc] is not supported by NPU currently. Now this kernel is running on CPU.");
    const auto self_cpu = self.to("cpu");
    auto result_cpu = result.to("cpu");
    at::sinc_out(result_cpu, self_cpu);
    result.copy_(result_cpu);
    return result;
}

at::Tensor sinc(const at::Tensor& self)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [sinc] is not supported by NPU currently. Now this kernel is running on CPU.");
    const auto self_cpu = self.to("cpu");
    at::sinc(self_cpu);
    return at::sinc(self_cpu).to(self.device());
}

at::Tensor& sinc_(at::Tensor& self)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [sinc] is not supported by NPU currently. Now this kernel is running on CPU.");
    auto self_cpu = self.to("cpu");
    at::sinc_(self_cpu);
    self.copy_(self_cpu);
    return self;
}

} // namespace acl_op
