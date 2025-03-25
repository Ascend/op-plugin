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
using npu_op_command = at_npu::native::OpCommand;

namespace {
at::Tensor& not_out_npu(at::Tensor& result, const at::Tensor& self)
{
    npu_op_command cmd;
    cmd.Name("LogicalNot")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& not_out_npu(at::Tensor& result, const at::Scalar self)
{
    npu_op_command cmd;
    cmd.Name("LogicalNot")
        .Input(self, self.type())
        .Output(result)
        .Run();
    return result;
}

at::Tensor& and_out_npu(at::Tensor& result, const at::Tensor& self, const at::Tensor& other)
{
    npu_op_command cmd;
    cmd.Name("LogicalAnd")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& and_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar& other)
{
    npu_op_command cmd;
    cmd.Name("LogicalAnd")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}

at::Tensor& or_out_npu(at::Tensor& result, const at::Tensor& self, const at::Scalar& other)
{
    npu_op_command cmd;
    cmd.Name("LogicalOr")
        .Input(self)
        .Input(other, self.scalar_type())
        .Output(result)
        .Run();
    return result;
}

at::Tensor& xor_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    if (self.dtype() == at::ScalarType::Bool) {
        auto not_self_result = npu_preparation::apply_tensor(self, output_size);
        not_out_npu(not_self_result, self);

        auto not_other_result = npu_preparation::apply_tensor(self, output_size);
        not_out_npu(not_other_result, other);

        auto not_self_and_other = npu_preparation::apply_tensor(self, output_size);
        and_out_npu(not_self_and_other, not_self_result, other);

        auto self_and_not_other = npu_preparation::apply_tensor(self, output_size);
        and_out_npu(self_and_not_other, self, not_other_result);

        npu_op_command cmd;
        cmd.Name("LogicalOr")
            .Input(not_self_and_other)
            .Input(self_and_not_other)
            .Output(result)
            .Run();
    } else {
        npu_op_command cmd;
        cmd.Name("BitwiseXor")
            .Input(self)
            .Input(other)
            .Output(result)
            .Run();
    }
    return result;
}

at::Tensor& xor_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& other)
{
    if (self.dtype() == at::ScalarType::Bool) {
        auto not_self_result = npu_preparation::apply_tensor(self);
        not_out_npu(not_self_result, self);

        auto not_self_or_other = npu_preparation::apply_tensor(self);
        or_out_npu(not_self_or_other, not_self_result, other);

        auto not_not_self_or_other = npu_preparation::apply_tensor(self);
        not_out_npu(not_not_self_or_other, not_self_or_other);

        auto not_self_and_other = npu_preparation::apply_tensor(self);
        and_out_npu(not_self_and_other, not_self_result, other);

        npu_op_command cmd;
        cmd.Name("LogicalOr")
            .Input(not_self_and_other)
            .Input(not_not_self_or_other)
            .Output(result)
            .Run();
    } else {
        npu_op_command cmd;
        cmd.Name("BitwiseXor")
            .Input(self)
            .Input(other, self.scalar_type())
            .Output(result)
            .Run();
    }
    return result;
}
} // namespace

at::Tensor __xor__(const at::Tensor& self, const at::Tensor& other)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    xor_out_npu(result, self, other);
    return result;
}

at::Tensor __xor__(const at::Tensor& self, const at::Scalar& other)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    xor_out_npu(result, self, other);
    return result;
}
}  // namespace acl_op
