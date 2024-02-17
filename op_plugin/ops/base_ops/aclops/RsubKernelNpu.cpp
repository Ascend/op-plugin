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

namespace {
at::Tensor rsub_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);

    return is_self_wrapped ? other : self;
}

at::Tensor &rsub_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other, at::Scalar alpha)
{
    // other*alpha
    at::Tensor other_mul_result;
    if (!op_plugin::utils::is_scalar_one(alpha)) {
        other_mul_result = at::mul(self, alpha);
    }

    at_npu::native::OpCommand cmd;
    if (other_mul_result.defined()) {
        cmd.Name("Sub").Input(other).Input(other_mul_result).Output(result).Run();
    } else {
        cmd.Name("Sub").Input(other).Input(self).Output(result).Run();
    }

    return result;
}

at::Tensor &rsub_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, at::Scalar other, at::Scalar alpha)
{
    // other*alpha
    at::Tensor scalar_value(at::mul(self, alpha));

    at_npu::native::OpCommand cmd;
    cmd.Name("Sub").Input(other, self.scalar_type()).Input(scalar_value).Output(result).Run();

    return result;
}
} // namespace

at::Tensor rsub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    at::Tensor output_tensor = rsub_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(output_tensor, output_size);
    rsub_out_npu_nocheck(result, self, other, alpha);

    return result;
}

at::Tensor rsub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    rsub_out_npu_nocheck(result, self, other, alpha);

    return result;
}
} // namespace acl_op
