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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& bitwise_and_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnBitwiseAndScalar, acl_op::bitwise_and_out(self, other, out));
    npu_preparation::check_tensor({self}, out, out, self.sizes());
    EXEC_NPU_CMD(aclnnBitwiseAndScalar, self, other, out);
    return out;
}

static at::Tensor& bitwise_and_op_api_out_npu_nocheck(at::Tensor& result, const at::Tensor& self,
                                                      const at::Tensor& other)
{
    if (!torch_npu::utils::is_npu(other)) {
        const at::Scalar other_value = other.item();
        EXEC_NPU_CMD(aclnnBitwiseAndScalar, self, other_value, result);
    } else if (!torch_npu::utils::is_npu(self)) {
        const at::Scalar self_value = self.item();
        EXEC_NPU_CMD(aclnnBitwiseAndScalar, other, self_value, result);
    } else {
        EXEC_NPU_CMD(aclnnBitwiseAndTensor, self, other, result);
    }
    return result;
}

at::Tensor& bitwise_and_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnBitwiseAndScalar, acl_op::bitwise_and_out(self, other, out));
    DO_COMPATIBILITY(aclnnBitwiseAndTensor, acl_op::bitwise_and_out(self, other, out));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self}, out, out, output_size);
    bitwise_and_op_api_out_npu_nocheck(out, self, other);
    return out;
}

at::Tensor bitwise_and(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnBitwiseAndScalar, acl_op::bitwise_and(self, other));
    DO_COMPATIBILITY(aclnnBitwiseAndTensor, acl_op::bitwise_and(self, other));

    if (!torch_npu::utils::is_npu(other)) {
        const at::Scalar other_value = other.item();
        return op_api::bitwise_and(self, other_value);
    }

    if (!torch_npu::utils::is_npu(self)) {
        const at::Scalar self_value = self.item();
        return op_api::bitwise_and(other, self_value);
    }

    // calculate the output size
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);

    at::Tensor ref_tensor;
    if (isSelfWrapped) {
        ref_tensor = other;
    } else {
        ref_tensor = self;
    }

    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

    // construct the output at::Tensor of the NPU
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnBitwiseAndTensor, self, other, result);
    return result;
}

at::Tensor bitwise_and(const at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnBitwiseAndScalar, acl_op::bitwise_and(self, other));
    // calculate the output size
    auto output_size = op_infer::input_same_output_size(self);

    // construct the output at::Tensor of the NPU
    at::Tensor result;
    if ((self.scalar_type() == at::ScalarType::Bool) && (!other.isBoolean())) {
        result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kLong));
    } else {
        result = npu_preparation::apply_tensor_without_format(self);
    }

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnBitwiseAndScalar, self, other, result);
    return result;
}

at::Tensor& bitwise_and_inplace_op_api_out_npu_nocheck(at::Tensor& self, const at::Tensor& other)
{
    if (!torch_npu::utils::is_npu(other)) {
        const at::Scalar other_value = other.item();
        EXEC_NPU_CMD(aclnnInplaceBitwiseAndScalar, self, other_value);
    } else {
        EXEC_NPU_CMD(aclnnInplaceBitwiseAndTensor, self, other);
    }
    return self;
}

at::Tensor& bitwise_and_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceBitwiseAndScalar, acl_op::bitwise_and_(self, other));
    DO_COMPATIBILITY(aclnnInplaceBitwiseAndTensor, acl_op::bitwise_and_(self, other));
    bitwise_and_inplace_op_api_out_npu_nocheck(self, other);
    return self;
}

at::Tensor& bitwise_and_(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceBitwiseAndScalar, acl_op::bitwise_and_(self, other));
    EXEC_NPU_CMD(aclnnInplaceBitwiseAndScalar, self, other);
    return self;
}
}
