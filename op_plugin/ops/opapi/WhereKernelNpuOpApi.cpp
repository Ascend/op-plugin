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

vector<at::Tensor> where(const at::Tensor &condition)
{
    return at::native::where(condition);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unify_tensors_to_npu(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other)
{
    std::vector<at::Tensor> tensors = {condition, self, other};
    c10::Device target_device = condition.device();
    bool found_npu = false;
    for (const auto& tensor : tensors) {
        if (!npu_preparation::is_scalar_wrapped_to_tensor(tensor)) {
            target_device = tensor.device();
            found_npu = true;
            break;
        }
    }

    if (!found_npu) {
        TORCH_NPU_WARN_ONCE("where does not have input tensors on NPU device, please check!");
        return std::make_tuple(condition, self, other);
    }

    at::Tensor condition_npu = condition;
    at::Tensor self_npu = self;
    at::Tensor other_npu = other;

    at::Scalar scalar;
    if (npu_preparation::is_scalar_wrapped_to_tensor(condition)) {
        scalar = condition.item();
        condition_npu = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, condition.scalar_type(), target_device);
    }
    if (npu_preparation::is_scalar_wrapped_to_tensor(self)) {
        scalar = self.item();
        self_npu = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, self.scalar_type(), target_device);
    }
    if (npu_preparation::is_scalar_wrapped_to_tensor(other)) {
        scalar = other.item();
        other_npu = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(), target_device);
    }
    return std::make_tuple(condition_npu, self_npu, other_npu);
}


at::Tensor& where_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnSWhere, acl_op::where_out(condition, self, other, out));

    auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);

    npu_preparation::check_tensor({condition, self, other}, out, out, output_size);

    auto [cond_cp, self_cp, other_cp] = unify_tensors_to_npu(condition, self, other);
    EXEC_NPU_CMD(aclnnSWhere, cond_cp, self_cp, other_cp, out);

    return out;
}

at::Tensor where(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnSWhere, acl_op::where(condition, self, other));
    auto broadcast_output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(condition.sizes(), broadcast_output_size);
    auto result_type = at::native::result_type(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    auto [cond_cp, self_cp, other_cp] = unify_tensors_to_npu(condition, self, other);
    EXEC_NPU_CMD(aclnnSWhere, cond_cp, self_cp, other_cp, result);

    return result;
}
}