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

at::Tensor& upsample_bilinear2d_backward_old_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                     acl_op::upsample_bilinear2d_backward_out(grad_output, output_size, input_size,
                                                              align_corners, scales_h, scales_w, grad_input));
    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, output_size, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor upsample_bilinear2d_backward_old(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                     acl_op::upsample_bilinear2d_backward(grad_output, output_size, input_size,
                                                          align_corners, scales_h, scales_w));
    auto outputSize = input_size;
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, outputSize);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, output_size, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor upsample_bilinear2d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackwardV2,
                     op_api::upsample_bilinear2d_backward_old(grad_output, output_size, input_size,
                                                              align_corners, scales_h, scales_w));
    auto outputSize = input_size;
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, outputSize);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackwardV2, grad_output, output_size, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor& upsample_bilinear2d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackwardV2,
                     op_api::upsample_bilinear2d_backward_old_out(grad_output, output_size, input_size,
                                                                  align_corners, scales_h, scales_w, grad_input));
    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackwardV2, grad_output, output_size, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_bilinear2d_backward(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackwardV2,
                     op_api::upsample_bilinear2d_backward_old(grad_output, output_size, input_size,
                                                              align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    auto outputsize = at::IntArrayRef(osize);
    auto outputSize = input_size;
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, outputSize);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackwardV2, grad_output, outputsize, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor upsample_bilinear2d_backward_old(
    const at::Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleBilinear2dBackward,
                     acl_op::upsample_bilinear2d_backward(grad_output, output_size, input_size,
                                                          align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(input_size, output_size, scale_factors);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 1);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    auto outputsize = at::IntArrayRef(osize);
    auto outputSize = input_size;
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output, outputSize);

    EXEC_NPU_CMD(aclnnUpsampleBilinear2dBackward, grad_output, outputsize, input_size, align_corners,
                 scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}
#endif
}
