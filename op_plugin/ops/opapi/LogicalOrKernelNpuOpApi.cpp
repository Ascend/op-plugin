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

at::Tensor& logical_or_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnLogicalOr, acl_op::logical_or_out(self, other, result));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self, other}, result, output_size);
    EXEC_NPU_CMD(aclnnLogicalOr, self, other, result);
    return result;
}

at::Tensor logical_or(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnLogicalOr, acl_op::logical_or(self, other));
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        auto cp_other = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(),
                                                                             self.device());
        EXEC_NPU_CMD(aclnnLogicalOr, self, cp_other, result);
    } else {
        EXEC_NPU_CMD(aclnnLogicalOr, self, other, result);
    }
    return result;
}

at::Tensor& logical_or_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceLogicalOr, acl_op::logical_or_(self, other));
    npu_preparation::check_memory({self, other},{self});
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        auto cp_other = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(),
                                                                             self.device());
        EXEC_NPU_CMD(aclnnInplaceLogicalOr, self, cp_other);
    } else {
        EXEC_NPU_CMD(aclnnInplaceLogicalOr, self, other);
    }
    return self;
}

}
