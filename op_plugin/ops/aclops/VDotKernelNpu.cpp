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
at::Tensor& vdot_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Dot")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor& vdot_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    c10::SmallVector<int64_t, N> output_size = {};
    npu_preparation::CheckOut(
        {self, other},
        out,
        self,
        output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out);
        vdot_out_npu_nocheck(contiguous_out, self, other);
        npu_utils::format_fresh_view(out, contiguous_out);
    } else {
        vdot_out_npu_nocheck(out, self, other);
    }

    return out;
}

at::Tensor vdot(const at::Tensor& self, const at::Tensor& other)
{
    c10::SmallVector<int64_t, N> output_size = {};
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    vdot_out_npu_nocheck(result, self, other);
    return result;
}
} // op_plugin
