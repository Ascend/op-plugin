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

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

namespace {
using npu_preparation = at_npu::native::OpPreparation;
const int64_t FP4_IN_UINT8_NUM = 2;

at::Tensor npu_dtype_cast_impl_op_api(
    const at::Tensor& self,
    int64_t dtype,
    c10::optional<int64_t> input_dtype)
{
    // check if input dtype is valid
    int64_t input_dtype_tocheck = input_dtype.has_value() ? input_dtype.value() : static_cast<int64_t>(self.scalar_type());
    bool special_output_type = (dtype == static_cast<int64_t>(c10_npu::DType::HIFLOAT8) ||
                                dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
    at::SmallVector<int64_t, op_infer::SIZE> input_shape;
    at::SmallVector<int64_t, op_infer::SIZE> output_shape;
    int32_t input_dim = self.dim();
    int32_t index = 0;
    for (; index < input_dim - 1; ++index) {
        input_shape.push_back(self.size(index));
        output_shape.push_back(self.size(index));
    }
    if (input_dim > 0) {
        if (input_dtype_tocheck == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
            input_dtype_tocheck == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2)) {
            input_shape.push_back(self.size(index) * FP4_IN_UINT8_NUM);
        } else {
            input_shape.push_back(self.size(index));
        }
  
        // float4 shape check
        if (dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2) ||
            dtype == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)) {
            TORCH_CHECK(input_shape[index] % FP4_IN_UINT8_NUM == 0,
                        "The last dim input shape must be divisible by 2 if "
                        "output dtype is torch_npu.float4_e2m1 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
            output_shape.push_back(input_shape[index] / FP4_IN_UINT8_NUM);
        } else {
            output_shape.push_back(input_shape[index]);
        }
    }
    at::Tensor output_tensor;
    aclDataType output_acltype;
    if (special_output_type) {
        output_tensor = npu_preparation::apply_tensor_without_format(output_shape, c10::ScalarType::Byte);
        output_acltype = c10_npu::GetAclDataType(dtype);
    } else {
        output_acltype = c10_npu::GetAclDataType(dtype);
        at::ScalarType c10_scalar_dtype = npu_preparation::convert_to_scalar_type(output_acltype);
        output_tensor = npu_preparation::apply_tensor_without_format(output_shape, c10::dtype(c10_scalar_dtype));
    }
    aclDataType input_acltype = c10_npu::GetAclDataType(input_dtype_tocheck);
    TensorWrapper input_wrapper = {self, input_acltype};
    TensorWrapper output_wrapper = {output_tensor, output_acltype};
    EXEC_NPU_CMD(aclnnCast, input_wrapper, output_acltype, output_wrapper);
    return output_tensor;
}
} // namespace

at::Tensor npu_dtype_cast(
    const at::Tensor& self,
    int64_t dtype,
    c10::optional<int64_t> input_dtype)
{
    DO_COMPATIBILITY(aclnnCast, acl_op::npu_dtype_cast(self, dtype, input_dtype));
    return npu_dtype_cast_impl_op_api(self, dtype, input_dtype);
}

at::Tensor _npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype)
{
    DO_COMPATIBILITY(aclnnCast, acl_op::_npu_dtype_cast(self, dtype));
    if (self.dtype() == dtype) {
        return self.clone();
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnCast, self, dtype, result);
    return result;
}

at::Tensor npu_dtype_cast_backward(
    const at::Tensor& grad,
    at::ScalarType dtype,
    c10::optional<int64_t> grad_dtype,
    c10::optional<int64_t> input_dtype)
{
    grad.requires_grad_();
    int64_t input_dtype_tocheck = input_dtype.has_value() ? input_dtype.value() : static_cast<int64_t>(dtype);
    return at_npu::native::custom_ops::npu_dtype_cast(grad, input_dtype_tocheck, grad_dtype);
};
at::Tensor _npu_dtype_cast_backward(const at::Tensor& grad, at::ScalarType dtype)
{
    grad.requires_grad_();
    return at_npu::native::custom_ops::_npu_dtype_cast(grad, dtype);
};
} // namespace op_api
