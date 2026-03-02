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
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/custom_dtype/Init.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace c10_npu;

constexpr int ATTRS_DIM = 2;
constexpr int TENSORS_DIM = 4;
constexpr int OFFSET_H_INDEX = 2;
constexpr int OFFSET_W_INDEX = 3;

std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d_out(const at::Tensor &input, const at::Tensor &weight,
                                                             const at::Tensor &offset,
                                                             const c10::optional<at::Tensor> &bias,
                                                             at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                             at::IntArrayRef padding, at::IntArrayRef dilation,
                                                             int64_t groups, int64_t deformable_groups, bool modulated)
{
    TORCH_CHECK(input.dim() >= TENSORS_DIM, "input has to be more than 4D, but got Tensor of dimension ",
                input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= TENSORS_DIM, "weight has to be more than 4D, but got Tensor of dimension ",
                weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(offset.dim() >= TENSORS_DIM, "offset has to be more than 4D, but got Tensor of dimension ",
                offset.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(kernel_size.size() >= ATTRS_DIM, "kernel_size has to contain more than 2 elements, but got ",
                kernel_size.size(), OPS_ERROR(ErrCode::PARAM));

    int64_t n = input.size(0);
    int64_t ci = input.size(1);
    int64_t co = weight.size(0);
    int64_t ho = offset.size(OFFSET_H_INDEX);
    int64_t wo = offset.size(OFFSET_W_INDEX);
    int64_t kh = kernel_size[0];
    int64_t kw = kernel_size[1];

    c10::SmallVector<int64_t, SIZE> deformable_offset_size = {n, ci, ho * kh, wo * kw};
    c10::SmallVector<int64_t, SIZE> conv_output_size = {n, co, ho, wo};

    auto deformable_offset = npu_preparation::apply_tensor_without_format(deformable_offset_size,
        input.options().dtype(input.dtype()));
    auto conv_output = npu_preparation::apply_tensor_without_format(conv_output_size,
        input.options().dtype(input.dtype()));

    EXEC_NPU_CMD(aclnnDeformableConv2d, input, weight, offset, bias, kernel_size, stride, padding, dilation,
        groups, deformable_groups, modulated, conv_output, deformable_offset);

    return std::make_tuple(conv_output, deformable_offset);
}

std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d(const at::Tensor &input, const at::Tensor &weight,
                                                         const at::Tensor &offset,
                                                         const c10::optional<at::Tensor> &bias,
                                                         at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                         at::IntArrayRef padding, at::IntArrayRef dilation,
                                                         int64_t groups, int64_t deformable_groups, bool modulated)
{
    // If aclnn interface is not implemented, call aclop
    DO_COMPATIBILITY(aclnnDeformableConv2d,
                     acl_op::npu_deformable_conv2d(input, weight, offset, bias, kernel_size, stride, padding, dilation,
                                                   groups, deformable_groups, modulated));

    if (c10_npu::IsAclnnOnly()) {
        return npu_deformable_conv2d_out(input, weight, offset, bias, kernel_size, stride, padding, dilation,
                                         groups, deformable_groups, modulated);
    } else {
        TORCH_NPU_WARN("current soc not support aclnnDeformableConv2d");
        return acl_op::npu_deformable_conv2d(input, weight, offset, bias, kernel_size, stride, padding, dilation,
                                             groups, deformable_groups, modulated);
    }
}

} // namespace op_api