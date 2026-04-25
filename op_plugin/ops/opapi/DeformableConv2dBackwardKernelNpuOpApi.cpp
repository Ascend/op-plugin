// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/custom_dtype/Init.h"

namespace op_api
{
    using namespace c10_npu;
    using npu_preparation = at_npu::native::OpPreparation;

    constexpr int ATTRS_DIM = 2;
    constexpr int TENSORS_DIM = 4;

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deformable_conv2dbk_out(
        const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &offset_out,
        const at::Tensor &weight, const at::Tensor &offset, at::IntArrayRef kernel_size, at::IntArrayRef stride,
        at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated)
    {
        TORCH_CHECK(input.dim() >= TENSORS_DIM, "The dim of input has to be more than 4D, but got Tensor of dimension ",
                    input.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(grad_output.dim() >= TENSORS_DIM, "The dim of grad_output has to be more than 4D, but got Tensor of dimension ",
                    grad_output.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(offset_out.dim() >= TENSORS_DIM, "The dim of offset_out has to be more than 4D, but got Tensor of dimension ",
                    offset_out.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(weight.dim() >= TENSORS_DIM, "The dim of weight has to be more than 4D, but got Tensor of dimension ",
                    weight.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(offset.dim() >= TENSORS_DIM, "The dim of offset has to be more than 4D, but got Tensor of dimension ",
                    offset.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(kernel_size.size() >= ATTRS_DIM, "The size of kernel_size has to contain more than 2 elements, but got ",
                    kernel_size.size(), OPS_ERROR(ErrCode::PARAM));

        c10::SmallVector<int64_t, SIZE> grad_bias_size = {grad_output.size(1)};
        auto grad_input = npu_preparation::apply_tensor_without_format(input.sizes(), input.options());
        auto grad_weight = npu_preparation::apply_tensor_without_format(weight.sizes(), weight.options());
        auto grad_offset = npu_preparation::apply_tensor_without_format(offset.sizes(), offset.options());
        auto grad_bias = npu_preparation::apply_tensor_without_format(grad_bias_size, grad_output.options());
        EXEC_NPU_CMD(aclnnDeformableConv2dBackward, input, grad_output, offset_out, weight, offset,
                     kernel_size, stride, padding, dilation, groups, deformable_groups, modulated,
                     grad_input, grad_weight, grad_offset, grad_bias);
        return std::make_tuple(grad_input, grad_weight, grad_offset, grad_bias);
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deformable_conv2dbk(
        const at::Tensor &input, const at::Tensor &grad_output, const at::Tensor &offset_out,
        const at::Tensor &weight, const at::Tensor &offset, at::IntArrayRef kernel_size, at::IntArrayRef stride,
        at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated)
    {
        // If aclnn interface is not implemented, call aclop
        DO_COMPATIBILITY(aclnnDeformableConv2dBackward,
                         acl_op::npu_deformable_conv2dbk(input, grad_output, offset_out, weight, offset,
                                                         kernel_size, stride, padding, dilation, groups, deformable_groups, modulated));

        if (c10_npu::IsAclnnOnly())
        {
            return npu_deformable_conv2dbk_out(input, grad_output, offset_out, weight, offset,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
        }
        else
        {
            return acl_op::npu_deformable_conv2dbk(input, grad_output, offset_out, weight, offset,
                                                   kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
        }
    }

} // namespace op_api
