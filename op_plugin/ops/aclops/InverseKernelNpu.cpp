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
at::Tensor& inverse_out_npu_nocheck(at::Tensor& result, const at::Tensor& self)
{
    at::Tensor self_cast = self;
    at::Tensor result_cast = result;
    if (self.scalar_type() == at::kHalf) {
        self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);
    }

    if (result.scalar_type() == at::kHalf) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, at::kFloat);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("MatrixInverse")
        .Input(self_cast)
        .Output(result_cast)
        .Attr("adjoint", false)
        .Run();
    result.copy_(result_cast);
    return result;
}
} // namespace

at::Tensor& inverse_out(
    const at::Tensor& self,
    at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        inverse_out_npu_nocheck(contiguous_result, self);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        inverse_out_npu_nocheck(out, self);
    }
    return out;
}

at::Tensor inverse(const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    inverse_out_npu_nocheck(result, self);
    return result;
}
} // namespace acl_op
