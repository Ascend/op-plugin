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

/**
 * Different from Pytorch1.11 for torch.floor_divide() using truncation division,
 * hostapi are corrected to use floor division.
 */
static at::Tensor& floor_divide_out_npu_opapi(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    // executing the NPU operator
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnFloorDivides, self, others, result);
    } else {
        EXEC_NPU_CMD(aclnnFloorDivide, self, other, result);
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

static at::Tensor& inplace_floor_divide_out_npu_opapi(at::Tensor& self, const at::Tensor& other)
{
    // executing the NPU operator
    if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
        c10::Scalar others = other.item();
        EXEC_NPU_CMD(aclnnInplaceFloorDivides, self, others);
    } else {
        EXEC_NPU_CMD(aclnnInplaceFloorDivide, self, other);
    }
    return self;
}

at::Tensor& floor_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnFloorDivides, acl_op::floor_divide_out(self, other, out));
    DO_COMPATIBILITY(aclnnFloorDivide, acl_op::floor_divide_out(self, other, out));
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    // calculate the output size
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType result_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, result_type, out.device());
    npu_preparation::check_tensor({self, other}, out, out.scalar_type(), output_size);

    // calculate the output result of the NPU
    floor_divide_out_npu_opapi(self_cp, other, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor floor_divide(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnFloorDivides, acl_op::floor_divide(self, other));
    DO_COMPATIBILITY(aclnnFloorDivide, acl_op::floor_divide(self, other));
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    // calculate the output size
    bool isSelfWrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    at::Tensor outputTensor = isSelfWrapped ? other : self;
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor self_cp = self_tensor_to_device(self, high_type, outputTensor.device());

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, outputTensor.options().dtype(high_type));

    // calculate the output result of the NPU
    floor_divide_out_npu_opapi(self_cp, other, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_names);
    return result;
}

at::Tensor floor_divide(const at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnFloorDivides, acl_op::floor_divide(self, other));
    auto outputSize = op_infer::input_same_output_size(self);
    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor result = npu_preparation::apply_tensor_without_format(outputSize, self.options().dtype(high_type));
    EXEC_NPU_CMD(aclnnFloorDivides, self, other, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor& floor_divide_(at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnInplaceFloorDivides, acl_op::floor_divide_(self, other));
    DO_COMPATIBILITY(aclnnInplaceFloorDivide, acl_op::floor_divide_(self, other));
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);

    npu_preparation::CheckMemory({self, other}, {self});
    inplace_floor_divide_out_npu_opapi(self, other);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}

at::Tensor& floor_divide_(at::Tensor& self, const at::Scalar& other)
{
    DO_COMPATIBILITY(aclnnInplaceFloorDivides, acl_op::floor_divide_(self, other));
    EXEC_NPU_CMD(aclnnInplaceFloorDivides, self, other);
    return self;
}

}
