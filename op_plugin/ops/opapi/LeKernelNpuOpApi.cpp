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

at::Tensor &le_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLeTensor, acl_op::le_out(self, other, result));
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);
    EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
    return result;
}

at::Tensor &le_out(const at::Tensor &self, const at::Scalar &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnLeScalar, acl_op::le_out(self, other, result));
    auto outputSize = self.sizes();
    at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);

    EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
    return result;
}

at::Tensor le(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnLeTensor, acl_op::le(self, other));
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        const at::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnLeScalar, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnLeTensor, self, other, result);
    }
    return result;
}

at::Tensor le(const at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnLeScalar, acl_op::le(self, other));
    auto outputSize = op_infer::input_same_output_size(self);
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(outputSize, self.options().dtype(at::kBool));
    EXEC_NPU_CMD(aclnnLeScalar, self, other, result);
    return result;
}

at::Tensor &le_(at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnInplaceLeScalar, acl_op::le_(self, other));
    EXEC_NPU_CMD(aclnnInplaceLeScalar, self, other);
    return self;
}

at::Tensor &le_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceLeTensor, acl_op::le_(self, other));
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        return op_api::le_(self, other.item());
    } else {
        TORCH_CHECK(self.device() == other.device(),
                    "Expected all tensors to be on the same device, but found at least two devices", OPS_ERROR(ErrCode::INTERNAL));
        at_npu::native::OpPreparation::CheckMemory({self, other}, {self});
        EXEC_NPU_CMD(aclnnInplaceLeTensor, self, other);
        return self;
    }
}

}
