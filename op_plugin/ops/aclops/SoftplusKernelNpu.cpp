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

at::Tensor& softplus_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SoftplusV2")
        .Input(self)
        .Output(result)
        .Attr("beta", beta)
        .Attr("threshold", threshold)
        .Run();
    return result;
}
} // namespace

at::Tensor& softplus_out(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        softplus_out_nocheck(contiguous_result, self, beta, threshold);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        softplus_out_nocheck(out, self, beta, threshold);
    }
    return out;
}

at::Tensor softplus(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    softplus_out_nocheck(result, self, beta, threshold);
    return result;
}

} // namespace acl_op
