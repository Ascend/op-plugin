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

inline void alpha_check_npu_tensor(const at::ScalarType self_dtype, const at::ScalarType other_dtype, at::Scalar alpha)
{
    TORCH_CHECK(isFloatingType(self_dtype) || isComplexType(self_dtype) ||
                isFloatingType(other_dtype) || isComplexType(other_dtype) || alpha.isIntegral(true),
                "For integral input tensors, argument alpha must not be a floating point number.",
                OPS_ERROR(ErrCode::TYPE));
}

inline void alpha_check_npu_scalar(const at::ScalarType self_dtype, at::Scalar other, at::Scalar alpha)
{
    TORCH_CHECK(isFloatingType(self_dtype) || isComplexType(self_dtype) ||
                other.isFloatingPoint() || alpha.isIntegral(true),
                "For integral input tensors, argument alpha must not be a floating point number.",
                OPS_ERROR(ErrCode::TYPE));
}

static at::Tensor &sub_out_npu_nocheck(const at::Tensor &self, const at::Tensor &other, const at::Scalar alpha,
                                       at::Tensor &result)
{
    if (npu_preparation::IsCPUScalar(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnSubs, self, other_scalar, alpha, result);
    } else {
        EXEC_NPU_CMD(aclnnSub, self, other, alpha, result);
    }
    return result;
}

static at::Tensor& inplace_sub_out_npu_no_check(at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha)
{
    if (npu_preparation::IsCPUScalar(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnInplaceSubs, self, other_scalar, alpha);
    } else {
        EXEC_NPU_CMD(aclnnInplaceSub, self, other, alpha);
    }
    return self;
}

static at::Tensor self_tensor_to_device(const at::Tensor &tensor, const at::ScalarType result_type,
                                        const c10::Device device)
{
    if (npu_preparation::is_scalar_wrapped_to_tensor(tensor)) {
        at::Scalar scalar = tensor.item();
        return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
    }
    return tensor;
}

static at::Tensor sub_dest_output(const at::Tensor& self, const at::Tensor& other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return is_self_wrapped ? other : self;
}

at::Tensor &sub_out(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnSub, acl_op::sub_out(self, other, alpha, result));
    DO_COMPATIBILITY(aclnnSubs, acl_op::sub_out(self, other, alpha, result));
    alpha_check_npu_tensor(self.scalar_type(), other.scalar_type(), alpha);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_converted = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self}, result, result, output_size);
    npu_preparation::check_memory({self, other}, {result});
    sub_out_npu_nocheck(self_converted, other, alpha, result);
    return result;
}

at::Tensor sub(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnSub, acl_op::sub(self, other, alpha));
    DO_COMPATIBILITY(aclnnSubs, acl_op::sub(self, other, alpha));
    alpha_check_npu_tensor(self.scalar_type(), other.scalar_type(), alpha);
    at::Tensor output_tensor = sub_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_converted = self_tensor_to_device(self, result_type, output_tensor.device());
    auto result = npu_preparation::apply_tensor_without_format(output_size, output_tensor.options().dtype(result_type));
    sub_out_npu_nocheck(self_converted, other, alpha, result);
    return result;
}

at::Tensor sub(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnSubs, acl_op::sub(self, other, alpha));
    alpha_check_npu_scalar(self.scalar_type(), other, alpha);
    auto output_size = op_infer::input_same_output_size(self);
    at::ScalarType result_type = at::native::result_type(self, other);
    auto result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    EXEC_NPU_CMD(aclnnSubs, self, other, alpha, result);
    return result;
}

at::Tensor &sub_(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnInplaceSub, acl_op::sub_(self, other, alpha));
    DO_COMPATIBILITY(aclnnInplaceSubs, acl_op::sub_(self, other, alpha));
    alpha_check_npu_tensor(self.scalar_type(), other.scalar_type(), alpha);
    npu_preparation::check_memory({self, other}, {self});
    inplace_sub_out_npu_no_check(self, other, alpha);
    return self;
}

at::Tensor &sub_(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnInplaceSubs, acl_op::sub_(self, other, alpha));
    alpha_check_npu_scalar(self.scalar_type(), other, alpha);
    EXEC_NPU_CMD(aclnnInplaceSubs, self, other, alpha);
    return self;
}
}
