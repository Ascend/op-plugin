// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& log10_out_npu_nocheck(at::Tensor& result, const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Log")
        .Input(self)
        .Output(result)
        .Attr("base", static_cast<float>(10.0))
        .Attr("scale", static_cast<float>(1.0))
        .Attr("shift", static_cast<float>(0.0))
        .Run();
    return result;
}
} // namespace

at::Tensor& log10_out(const at::Tensor& self, at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        log10_out_npu_nocheck(contiguous_result, self);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        log10_out_npu_nocheck(out, self);
    }
    return out;
}

at::Tensor log10(const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    log10_out_npu_nocheck(result, self);
    return result;
}

at::Tensor& log10_(at::Tensor& self)
{
    return acl_op::log10_out(self, self);
}
} // namespace acl_op
