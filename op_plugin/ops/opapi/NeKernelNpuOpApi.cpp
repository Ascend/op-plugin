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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnNeTensor, acl_op::ne_out(self, other, result));
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self, other}, result, result.scalar_type(), at::IntArrayRef(outputSize));
    EXEC_NPU_CMD(aclnnNeTensor, self, other, result);
    return result;
}

at::Tensor& ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnNeScalar, acl_op::ne_out(self, other, result));
    npu_preparation::check_tensor({self}, result, result.scalar_type(), self.sizes());
    EXEC_NPU_CMD(aclnnNeScalar, self, other, result);
    return result;
}

at::Tensor ne(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnNeTensor, acl_op::ne(self, other));
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));

    if (npu_preparation::IsCPUScalar(other)) {
        const at::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnNeScalar, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnNeTensor, self, other, result);
    }
    return result;
}

at::Tensor ne(const at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnNeScalar, acl_op::ne(self, other));
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kBool));

    EXEC_NPU_CMD(aclnnNeScalar, self, other, result);
    return result;
}

at::Tensor& ne_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceNeTensor, acl_op::ne_(self, other));
    npu_preparation::check_memory({self, other}, {self});
    if (npu_preparation::IsCPUScalar(other)) {
        return op_api::ne_(self, other.item());
    } else {
        EXEC_NPU_CMD(aclnnInplaceNeTensor, self, other);
        return self;
    }
}

at::Tensor& ne_(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceNeScalar, acl_op::ne_(self, other));
    EXEC_NPU_CMD(aclnnInplaceNeScalar, self, other);
    return self;
}

}
