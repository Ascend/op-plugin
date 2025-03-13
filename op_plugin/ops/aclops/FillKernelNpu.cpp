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
at::Tensor& fill_out_nocheck(at::Tensor& result, at::Tensor& self, const at::Tensor& value)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Fill");
    if (self.dim() == 0) {
        c10::SmallVector<int64_t, N> dims = {1};
        cmd.Input(dims, at::kLong);
    } else {
        cmd.Input(self.sizes(), at::kLong);
    }
    cmd.Input(value)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& fill_out_nocheck(at::Tensor& result, at::Tensor& self, at::Scalar value)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Fill");
    if (self.dim() == 0) {
        c10::SmallVector<int64_t, N> dims = {1};
        cmd.Input(dims, at::kLong);
    } else {
        cmd.Input(self.sizes(), at::kLong);
    }
    cmd.Input(value, self.scalar_type(), at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& fill_out_nocheck(at::Tensor& self, const at::Tensor& value)
{
    if (npu_preparation::IsCPUScalar(value)) {
        fill_out_nocheck(self, self, value.item());
    } else {
        fill_out_nocheck(self, self, value);
    }
    return self;
}
} // namespace

at::Tensor& fill_(at::Tensor& self, const at::Tensor& value)
{
    auto value_dim = value.dim();
    TORCH_CHECK(value_dim <= 1, "fill_ only supports 0 or 1 dimension value tensor but got tensor with ",
        value_dim, " dimension." + OPS_ERROR(ErrCode::PARAM));
    npu_preparation::CheckMemory({self, value}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        fill_out_nocheck(contiguous_self, value);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        fill_out_nocheck(self, value);
    }
    return self;
}

at::Tensor& fill_(at::Tensor& self, const at::Scalar& value)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        fill_out_nocheck(contiguous_self, contiguous_self, value);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        fill_out_nocheck(self, self, value);
    }
    return self;
}
}  // namespace acl_op
