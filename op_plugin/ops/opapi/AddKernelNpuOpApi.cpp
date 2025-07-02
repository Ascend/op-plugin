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

inline void alpha_check_npu(const at::ScalarType dtype, at::Scalar alpha)
{
    TORCH_CHECK(!alpha.isBoolean() || dtype == at::ScalarType::Bool,
                "Boolean alpha only supported for Boolean results." + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype) || alpha.isIntegral(true),
                "For integral input tensors, argument alpha must not be a floating point number."
                + OPS_ERROR(ErrCode::TYPE));
}

static at::Tensor self_tensor_to_device(const at::Tensor &tensor, const at::ScalarType result_type,
                                        const c10::Device device)
{
    if (npu_preparation::is_scalar_wrapped_to_tensor(tensor) ||
        (tensor.dim() == 0 && !torch_npu::utils::is_npu(tensor))) {
        at::Scalar scalar = tensor.item();
        return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
    }
    return tensor;
}

static at::Tensor &add_out_npu_nocheck(
    const at::Tensor &self,
    const at::Tensor &other,
    const at::Scalar &alpha,
    at::Tensor &result)
{
    // executing the NPU operator
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnAdds, self, others, alpha, result);
    } else {
        if (self.dim() == 0 && !torch_npu::utils::is_npu(self)) {
            // self is a scalar.
            static const bool is_aclnn_available = check_aclnn_kernel_available("aclnnAddV3");
            if (is_aclnn_available) {
                c10::Scalar selfs = self.item();
                EXEC_NPU_CMD(aclnnAddV3, selfs, other, alpha, result);
            } else {
                at::Tensor self_cp = self_tensor_to_device(self, result.scalar_type(), result.device());
                EXEC_NPU_CMD(aclnnAdd, self_cp, other, alpha, result);
            }
        } else {
            // self and other are all npu tensors.
            EXEC_NPU_CMD(aclnnAdd, self, other, alpha, result);
        }
    }
    return result;
}

static at::Tensor &inplace_add_out_npu_no_check(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    // check if other scalar tensor
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnInplaceAdds, self, other_scalar, alpha);
    } else {
        EXEC_NPU_CMD(aclnnInplaceAdd, self, other, alpha);
    }
    return self;
}

static at::Tensor add_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return isSelfWrapped ? other : self;
}

at::Tensor add(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnAdd, acl_op::add(self, other, alpha));
    DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
    // calculate the output size
    at::Tensor output_tensor = add_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    alpha_check_npu(result_type, alpha);
    // construct the output tensor of the NPU
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, output_tensor.options().dtype(result_type));
    // calculate the output result of the NPU
    add_out_npu_nocheck(self, other, alpha, result);
    return result;
}

at::Tensor add(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnAdds, acl_op::add(self, other, alpha));
    // calculate the output size
    auto output_size = op_infer::input_same_output_size(self);
    at::ScalarType result_type = at::native::result_type(self, other);
    alpha_check_npu(result_type, alpha);
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAdds, self, other, alpha, result);
    return result;
}

at::Tensor &add_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnAdd, acl_op::add_out(self, other, alpha, result));
    DO_COMPATIBILITY(aclnnAdds, acl_op::add_out(self, other, alpha, result));
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor output_tensor = isSelfWrapped ? other : self;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);

    npu_preparation::check_tensor({self}, result, result, output_size);
    npu_preparation::check_memory({self, other}, {result});
    add_out_npu_nocheck(self, other, alpha, result);
    return result;
}

at::Tensor &add_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnInplaceAdd, acl_op::add_(self, other, alpha));
    DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
    npu_preparation::check_memory({self, other}, {self});
    inplace_add_out_npu_no_check(self, other, alpha);
    return self;
}

at::Tensor &add_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnInplaceAdds, acl_op::add_(self, other, alpha));
    EXEC_NPU_CMD(aclnnInplaceAdds, self, other, alpha);
    return self;
}
}
