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

at::Tensor isneginf(const at::Tensor &self)
{
    // convert args to cpu in order to use at kernel
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [isneginf] is not supported by NPU currently. Now this kernel is running on CPU.");
    const auto self_cpu = self.cpu();
    auto result = at::isneginf(self_cpu);
    return result.to(self.device());
}

at::Tensor& isneginf_out(const at::Tensor &self, at::Tensor &out)
{
    // convert args to cpu in order to use at kernel
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [isneginf.out] is not supported by NPU currently. Now this kernel is running on CPU.");
    const auto self_cpu = self.cpu();
    auto out_cpu = out.cpu();
    out_cpu = at::isneginf_out(out_cpu, self_cpu);
    out.copy_(out_cpu);
    return out;
}
}  // acl_op
