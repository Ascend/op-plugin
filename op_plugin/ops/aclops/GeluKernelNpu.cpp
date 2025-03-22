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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor gelu(const at::Tensor& self)
{
    return gelu_common_nocheck(self);
}

at::Tensor& gelu_out(const at::Tensor& self, at::Tensor& result)
{
    npu_preparation::CheckOut({self}, result, self);

    at_npu::native::OpCommand cmd;
    cmd.Name("Gelu")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor gelu(const at::Tensor& self, c10::string_view gelu)
{
    return gelu_common_nocheck(self);
}
#endif
} // namespace acl_op
