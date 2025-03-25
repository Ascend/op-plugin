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
at::Tensor& add_relu_out_nocheck(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha)
{
    at::Tensor add_result = acl_op::add(self, other, alpha);
    at_npu::native::OpCommand cmd;
    cmd.Name("Relu")
        .Input(add_result)
        .Output(out)
        .Run();
    return out;
}
} // namespace

at::Tensor& _add_relu_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self, other},
        out,
        self);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        add_relu_out_nocheck(contiguous_result, self, other, alpha);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        add_relu_out_nocheck(out, self, other, alpha);
    }
    return out;
}

at::Tensor _add_relu(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha)
{
    at::Tensor out = npu_preparation::apply_tensor(self);
    add_relu_out_nocheck(out, self, other, alpha);
    return out;
}

at::Tensor& _add_relu_(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha)
{
    return acl_op::_add_relu_out(self, other, alpha, self);
}
} // namespace acl_op
