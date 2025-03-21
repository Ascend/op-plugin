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
using npu_format_helper = at_npu::native::FormatHelper;

namespace {
at::Tensor& npu_broadcast_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef size)
{
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
    at::Tensor& result)
{
    npu_broadcast_out_nocheck(result, self, size);
    return result;
}

at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size)
{
    at::Tensor result;
    at::Tensor self_cp;
    if (self.dtype() == at::kBool) {
        self_cp = at_npu::native::custom_ops::npu_dtype_cast(self, at::kInt);
        if (!npu_format_helper::IsBaseFormatType(self_cp)) {
            auto format = npu_format_helper::GetBaseFormat(self_cp);
            at_npu::native::custom_ops::npu_format_cast_(self_cp, format);
        }
        result = npu_preparation::apply_tensor(self_cp, size);
        npu_broadcast_out_nocheck(result, self_cp, size);
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
    } else {
        if (!npu_format_helper::IsBaseFormatType(self)) {
            auto format = npu_format_helper::GetBaseFormat(self);
            self_cp = at_npu::native::custom_ops::npu_format_cast(self, format);
        } else {
            self_cp = self;
        }
        result = npu_preparation::apply_tensor(self_cp, size);
        npu_broadcast_out_nocheck(result, self_cp, size);
    }
    return result;
}
} // namespace acl_op
