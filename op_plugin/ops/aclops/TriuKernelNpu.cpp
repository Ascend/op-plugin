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
at::Tensor& triu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, int64_t diagonal)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Triu")
        .Input(self)
        .Output(result)
        .Attr("diagonal", diagonal)
        .Run();
    return result;
}
} // namespace

at::Tensor& triu_out(const at::Tensor& self, int64_t diagonal, at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        triu_out_npu_nocheck(contiguous_result, self, diagonal);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        triu_out_npu_nocheck(out, self, diagonal);
    }
    return out;
}

at::Tensor triu(const at::Tensor& self, int64_t diagonal)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    triu_out_npu_nocheck(result, self, diagonal);
    return result;
}

at::Tensor& triu_(at::Tensor& self, int64_t diagonal)
{
    return acl_op::triu_out(self, diagonal, self);
}
} // namespace acl_op
