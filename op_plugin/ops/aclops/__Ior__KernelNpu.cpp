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
at::Tensor &ior_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
    at_npu::native::OpCommand cmd;
    cmd.Name(real_op_name).Input(self).Input(other).Output(result).Run();
    return result;
}

at::Tensor &ior_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, at::Scalar other)
{
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
    at_npu::native::OpCommand cmd;
    cmd.Name(real_op_name).Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}
} // namespace

at::Tensor &__ior__(at::Tensor &self, const at::Tensor &other)
{
    npu_preparation::CheckMemory({self, other}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        ior_out_npu_nocheck(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        ior_out_npu_nocheck(self, self, other);
    }
    return self;
}

at::Tensor &__ior__(at::Tensor &self, const at::Scalar &other)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        ior_out_npu_nocheck(contiguous_self, contiguous_self, other);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        ior_out_npu_nocheck(self, self, other);
    }
    return self;
}
} // namespace acl_op
