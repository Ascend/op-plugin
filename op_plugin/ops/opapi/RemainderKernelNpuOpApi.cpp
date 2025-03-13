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

static const int64_t SIZE_INT64 = 8;

// get the shape result after broadcast
static at::Tensor remainder_dest_output(const at::Tensor& self, const at::Tensor& other)
{
    bool isSelfWrapped = !torch_npu::utils::is_npu(self);
    return isSelfWrapped ? other : self;
}

// tensor + scalar
at::Tensor& remainder_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnRemainderTensorScalar, acl_op::remainder_out(self, other, out));
    npu_preparation::check_tensor({self}, out, out.scalar_type(), self.sizes());
    EXEC_NPU_CMD(aclnnRemainderTensorScalar, self, other, out);
    return out;
}

at::Tensor remainder(const at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnRemainderTensorScalar, acl_op::remainder(self, other));
    at::ScalarType result_type = at::native::result_type(self, other); // promote_type
    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(result_type));
    EXEC_NPU_CMD(aclnnRemainderTensorScalar, self, other, result);
    return result;
}

at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceRemainderTensorScalar, acl_op::remainder_(self, other));
    EXEC_NPU_CMD(aclnnInplaceRemainderTensorScalar, self, other);
    return self;
}

// scalar + tensor
at::Tensor remainder(const at::Scalar& self, const at::Tensor& other)
{
    at::ScalarType result_type = at::native::result_type(self, other); // promote_type
    at::Tensor result = npu_preparation::apply_tensor(other, other.options().dtype(result_type));
    EXEC_NPU_CMD(aclnnRemainderScalarTensor, self, other, result);
    return result;
}

// tensor + tensor
at::Tensor& remainder_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnRemainderTensorTensor, acl_op::remainder_out(self, other, out));
    auto broadcast_shape = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self, other}, out, out.scalar_type(), broadcast_shape);

    EXEC_NPU_CMD(aclnnRemainderTensorTensor, self, other, out);
    return out;
}

at::Tensor remainder(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnRemainderTensorTensor, acl_op::remainder(self, other));
    at::Tensor output_tensor = remainder_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other); // promote_type
    at::Tensor result = npu_preparation::apply_tensor(output_size, output_tensor.options().dtype(result_type),
        output_tensor);
    EXEC_NPU_CMD(aclnnRemainderTensorTensor, self, other, result);

    return result;
}

at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceRemainderTensorTensor, acl_op::remainder_(self, other));
    at::ScalarType promote_type = at::native::result_type(self, other);
    EXEC_NPU_CMD(aclnnInplaceRemainderTensorTensor, self, other);

    return self;
}
}  // namespace op_api

