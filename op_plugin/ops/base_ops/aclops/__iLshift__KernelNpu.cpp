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
at::Tensor &ilshift_out_npu(at::Tensor &result, at::Tensor &self, at::Scalar other)
{
    at::Tensor other_tensor = npu_preparation::apply_tensor(self);
    at::Tensor other_broadcast = acl_op::fill_(other_tensor, other);
    at_npu::native::OpCommand cmd;
    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
    return result;
}

at::Tensor &ilshift_out_npu(at::Tensor &result, at::Tensor &self, const at::Tensor &other)
{
    at::Tensor other_broadcast = other.expand(self.sizes());
    at_npu::native::OpCommand cmd;
    cmd.Name("LeftShift").Input(self).Input(other_broadcast).Output(result).Run();
    return result;
}
} // namespace

at::Tensor &__ilshift__(at::Tensor &self, const at::Tensor &other)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        ilshift_out_npu(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        ilshift_out_npu(self, self, other);
    }
    return self;
}

at::Tensor &__ilshift__(at::Tensor &self, const at::Scalar &other)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        ilshift_out_npu(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        ilshift_out_npu(self, self, other);
    }
    return self;
}
} // namespace acl_op
