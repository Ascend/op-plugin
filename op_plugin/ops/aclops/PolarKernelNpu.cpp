// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/op_api_common.h"

namespace acl_op {

at::Tensor& polar_out(const at::Tensor& abs, const at::Tensor& angle, at::Tensor& out)
{
    TORCH_WARN_ONCE(
        "Warning: kernel [polar.out] is not supported by NPU currently. Now this kernel is running on CPU.");
    auto abs_cpu = abs.cpu();
    auto angle_cpu = angle.cpu();
    auto result_cpu = out.cpu();
    result_cpu = at::polar_out(result_cpu, abs_cpu, angle_cpu);

    out.copy_(result_cpu.to(out.device()));
    return out;
}

at::Tensor polar(const at::Tensor& abs, const at::Tensor& angle)
{
    TORCH_WARN_ONCE(
        "Warning: kernel [polar] is not supported by NPU currently. Now this kernel is running on CPU.");

    auto abs_cpu = abs.cpu();
    auto angle_cpu = angle.cpu();
    at::Tensor result_cpu = at::polar(abs_cpu, angle_cpu);

    at::Tensor result = result_cpu.to(abs.device());
    return result;
}

} // namespace op_api
