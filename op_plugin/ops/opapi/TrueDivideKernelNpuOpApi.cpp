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

#include "op_plugin/utils/op_api_common.h"

namespace op_api {
#if VERSION_BETWEEN(V2R0, V2R0)
using npu_preparation = at_npu::native::OpPreparation;

static at::Tensor& div_out_npu_opapi_nocheck(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    // executing the NPU operator
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnDivs, self, others, result);
    } else {
        EXEC_NPU_CMD(aclnnDiv, self, other, result);
    }
    return result;
}

static at::Tensor self_tensor_to_device(const at::Tensor& tensor, const at::ScalarType result_type,
                                        const c10::Device device)
{
    if (npu_preparation::is_scalar_wrapped_to_tensor(tensor)) {
        at::Scalar scalar = tensor.item();
        return npu_preparation::copy_scalar_to_device(scalar, result_type, device);
    }
    return tensor;
}

at::Tensor true_divide(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnDivs, acl_op::true_divide(self, other));
    DO_COMPATIBILITY(aclnnDiv, acl_op::true_divide(self, other));
    // calculate the output size
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor output_tensor = is_self_wrapped ? other : self;
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, high_type, output_tensor.device());

    if (isIntegralType(high_type, true)) {
        high_type = at::ScalarType::Float;
    }
    // construct the output tensor of the NPU
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, output_tensor.options().dtype(high_type));

    // calculate the output result of the NPU
    div_out_npu_opapi_nocheck(self_cp, other, result);
    return result;
}

at::Tensor true_divide(const at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnDivs, acl_op::true_divide(self, other));
    auto output_size = op_infer::input_same_output_size(self);
    at::ScalarType high_type = at::native::result_type(self, other);
    if (isIntegralType(high_type, true)) {
        high_type = at::ScalarType::Float;
    }
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(high_type));
    EXEC_NPU_CMD(aclnnDivs, self, other, result);
    return result;
}

at::Tensor& true_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnDivs, acl_op::true_divide_out(self, other, result));
    DO_COMPATIBILITY(aclnnDiv, acl_op::true_divide_out(self, other, result));
    // calculate the output size
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    if (isIntegralType(result_type, true)) {
        result_type = at::ScalarType::Float;
    }
    if (isFloatingType(result.scalar_type())) {
        result_type = result.scalar_type();
    }
    at::Tensor self_cp = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self, other}, result, result_type, output_size);

    // calculate the output result of the NPU
    div_out_npu_opapi_nocheck(self_cp, other, result);
    return result;
}

at::Tensor& true_divide_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceDiv, acl_op::true_divide_(self, other));
    npu_preparation::check_memory({self, other}, {self});

    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar other_value = other.item();
        true_divide_(self, other_value);
    } else {
        EXEC_NPU_CMD(aclnnInplaceDiv, self, other);
    }
    return self;
}

at::Tensor& true_divide_(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceDivs, acl_op::true_divide_(self, other));
    EXEC_NPU_CMD(aclnnInplaceDivs, self, other);
    return self;
}
#endif
}  // namespace op_api
