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

namespace {
at::Tensor& tanh_out_nocheck(at::Tensor& result, const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Tanh")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& tanh_out(const at::Tensor& self, at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        self);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        tanh_out_nocheck(contiguous_result, self);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        tanh_out_nocheck(result, self);
    }

    return result;
}

at::Tensor tanh(const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    tanh_out_nocheck(result, self);
    return result;
}

at::Tensor& tanh_(at::Tensor& self)
{
    return acl_op::tanh_out(self, self);
}
} // namespace acl_op
