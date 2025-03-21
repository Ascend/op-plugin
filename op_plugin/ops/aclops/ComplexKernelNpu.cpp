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
at::Tensor complex(const at::Tensor &real, const at::Tensor &imag)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [complex] is not supported by NPU currently. Now this kernel is running on CPU.");
    at::Tensor real_cpu = real.to("cpu");
    at::Tensor imag_cpu = imag.to("cpu");
    auto result = at::native::complex(real_cpu, imag_cpu);
    at::Tensor output = result.to(real.device());
    return output;
}

at::Tensor &complex_out(const at::Tensor &real, const at::Tensor &imag, at::Tensor &out)
{
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [complex_out] is not supported by NPU currently. Now this kernel is running on CPU.");
    at::Tensor real_cpu = real.to("cpu");
    at::Tensor imag_cpu = imag.to("cpu");
    at::Tensor out_cpu = out.to("cpu");
    at::native::complex_out(real_cpu, imag_cpu, out_cpu);
    // calculate the output size
    auto output_size = op_infer::broadcast_ops_npu_output_size(real, imag);
    out.resize_(output_size);
    out.copy_(out_cpu);
    return out;
}
} // namespace acl_op
