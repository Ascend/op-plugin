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
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

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

at::Tensor &inplace_mul_out_npu_no_check(at::Tensor &self, const at::Tensor &other)
{
    // check if other scalar tensor
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnInplaceMuls, self, other_scalar);
    } else {
        EXEC_NPU_CMD(aclnnInplaceMul, self, other);
    }
    return self;
}

at::Tensor &mul_out_npu_no_check(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    // check if other scalar tensor
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnMuls, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnMul, self, other, result);
    }
    return result;
}

static at::Tensor mul_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return isSelfWrapped ? other : self;
}

at::Tensor &mul_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnMul, acl_op::mul_out(self, other, result));
    DO_COMPATIBILITY(aclnnMuls, acl_op::mul_out(self, other, result));
    // calculate the output size
    at::Tensor output_tensor = mul_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self}, result, result.scalar_type(), output_size);
    // calculate the output result of the NPU
    mul_out_npu_no_check(self_cp, other, result);
    return result;
}

at::Tensor mul(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnMul, acl_op::mul(self, other));
    DO_COMPATIBILITY(aclnnMuls, acl_op::mul(self, other));
    // calculate the output size
    at::Tensor output_tensor = mul_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, output_tensor.device());

    // construct the output tensor of the NPU
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, output_tensor.options().dtype(result_type));

    // calculate the output result of the NPU
    mul_out_npu_no_check(self_cp, other, result);
    return result;
}

at::Tensor mul(const at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnMuls, acl_op::mul(self, other));
    auto output_size = op_infer::input_same_output_size(self);
    at::ScalarType result_type = at::native::result_type(self, other);
    // construct the output tensor of the Npu
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnMuls, self, other, result);
    return result;
}

at::Tensor &mul_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceMul, acl_op::mul_(self, other));
    DO_COMPATIBILITY(aclnnInplaceMuls, acl_op::mul_(self, other));
    TORCH_CHECK(torch_npu::utils::is_npu(self), "Inplace tensor self must be NPU-Tensor.", OPS_ERROR(ErrCode::PARAM));
    npu_preparation::check_memory({self, other}, {self});
    inplace_mul_out_npu_no_check(self, other);
    return self;
}

at::Tensor &mul_(at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnInplaceMuls, acl_op::mul_(self, other));
    TORCH_CHECK(torch_npu::utils::is_npu(self), "Inplace tensor self must be NPU-Tensor.", OPS_ERROR(ErrCode::PARAM));
    EXEC_NPU_CMD(aclnnInplaceMuls, self, other);
    return self;
}
}
