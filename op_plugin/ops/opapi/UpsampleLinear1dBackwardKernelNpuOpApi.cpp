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

at::Tensor upsample_linear1d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleLinear1dBackward,
                     acl_op::upsample_linear1d_backward(grad_output, output_size, input_size, align_corners, scales));
    double scales_attr = scales.value_or(0);

    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, input_size);

    EXEC_NPU_CMD(aclnnUpsampleLinear1dBackward, grad_output, output_size, input_size, align_corners,
                 scales_attr, grad_input);
    return grad_input;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_linear1d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleLinear1dBackward,
                     acl_op::upsample_linear1d_backward(grad_output, output_size, input_size,
                                                        align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto outputsize = at::IntArrayRef(osize);
    auto scales_l = op_plugin::utils::get_scale_value(scale_factors, 0);
    double scales_l_attr = scales_l.value_or(0);

    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, input_size);

    EXEC_NPU_CMD(aclnnUpsampleLinear1dBackward, grad_output, outputsize, input_size, align_corners,
                 scales_l_attr, grad_input);
    return grad_input;
}
#endif
}
