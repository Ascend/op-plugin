// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool wrapped_to_scalar(const at::Tensor &tensor)
{
    return tensor.dim() == 0 && (tensor.scalar_type() == at::ScalarType::Double ||
        tensor.scalar_type() == at::ScalarType::Long) && c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
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

at::Tensor &inplace_fmod_out_npu_no_check(at::Tensor &self, const at::Tensor &other)
{
    // wrapped tensor will turn to tensor+scalar process
    if (wrapped_to_scalar(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnInplaceFmodScalar, self, other_scalar);
    } else {
        EXEC_NPU_CMD(aclnnInplaceFmodTensor, self, other);
    }
    return self;
}

at::Tensor &fmod_out_npu_no_check(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    // wrapped tensor will turn to tensor+scalar process
    if (wrapped_to_scalar(other)) {
        c10::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnFmodScalar, self, other_scalar, result);
    } else {
        EXEC_NPU_CMD(aclnnFmodTensor, self, other, result);
    }
    return result;
}

static at::Tensor fmod_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return isSelfWrapped ? other : self;
}

at::Tensor &fmod_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnFmodTensor, acl_op::fmod_out(self, other, result));
    DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod_out(self, other, result));
    // calculate the output size
    at::Tensor output_tensor = fmod_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, result.device());
    npu_preparation::check_tensor({self}, result, result.scalar_type(), output_size);
    // calculate the output result of the NPU
    fmod_out_npu_no_check(self_cp, other, result);
    return result;
}

at::Tensor fmod(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnFmodTensor, acl_op::fmod(self, other));
    DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod(self, other));
    // calculate the output size
    at::Tensor output_tensor = fmod_dest_output(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, output_tensor.device());

    // construct the output tensor of the NPU
    at::Tensor result =
        npu_preparation::apply_tensor_without_format(output_size, output_tensor.options().dtype(result_type));

    // calculate the output result of the NPU
    fmod_out_npu_no_check(self_cp, other, result);
    return result;
}

at::Tensor &fmod_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceFmodTensor, acl_op::fmod_(self, other));
    DO_COMPATIBILITY(aclnnInplaceFmodScalar, acl_op::fmod_(self, other));
    TORCH_CHECK(torch_npu::utils::is_npu(self), "Inplace tensor self must be NPU-Tensor.", OPS_ERROR(ErrCode::PARAM));
    npu_preparation::check_memory({self, other}, {self});
    inplace_fmod_out_npu_no_check(self, other);
    return self;
}
}
