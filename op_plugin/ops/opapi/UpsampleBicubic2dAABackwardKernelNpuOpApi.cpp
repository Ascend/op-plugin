// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/custom_functions/opapi/UpsampleConstants.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

bool checkBicubicBackwardScales(float realScale_h, float realScale_w)
{
    if (realScale_h > 0.0 && realScale_h < BICUBIC_MIN_SCALE) {
        return false;
    }
    if (realScale_w > 0.0 && realScale_w < BICUBIC_MIN_SCALE) {
        return false;
    }
    return true;
}

bool checkBicubicBackwardShapes(int outputSize_h, int outputSize_w)
{
    return outputSize_h <= BICUBIC_MAX_SHAPE && outputSize_w <= BICUBIC_MAX_SHAPE;
}

bool checkBicubicBackwardUseFast(
    const at::Tensor &grad_output, bool align_corners, double scales_h, double scales_w, at::Tensor &grad_input)
{
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                              c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    double realScale_h =
        op_plugin::utils::compute_scale(grad_input.size(H_INDEX), grad_output.size(H_INDEX), scales_h);
    double realScale_w =
        op_plugin::utils::compute_scale(grad_input.size(W_INDEX), grad_output.size(W_INDEX), scales_w);
    if (!is_support_nd_out || !checkBicubicBackwardScales(realScale_h, realScale_w) ||
        !checkBicubicBackwardShapes(grad_output.size(H_INDEX), grad_output.size(W_INDEX))) {
        return false;
    }
    return true;
}

at::Tensor &upsample_bicubic2d_aa_backward_out_slow(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w,
    at::Tensor &grad_input)
{
    auto scalar_type = grad_output.scalar_type();
    at::Tensor grad_output_slow = grad_output.cpu().to(at::ScalarType::Float);

    at::Tensor grad_input_slow = at::_upsample_bicubic2d_aa_backward(
        grad_output_slow, output_size, input_size, align_corners, scales_h, scales_w);
    grad_input.copy_(grad_input_slow.to(scalar_type));
    return grad_input;
}

at::Tensor upsample_bicubic2d_aa_backward_slow(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    auto scalar_type = grad_output.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    at::Tensor grad_output_slow = grad_output.cpu().to(at::ScalarType::Float);

    at::Tensor grad_input_slow = at::_upsample_bicubic2d_aa_backward(
        grad_output_slow, output_size, input_size, align_corners, scales_h, scales_w);
    grad_input.copy_(grad_input_slow.to(scalar_type));
    return grad_input;
}

at::Tensor &_upsample_bicubic2d_aa_backward_out(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w,
    at::Tensor &grad_input)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2dAAGrad,
        upsample_bicubic2d_aa_backward_out_slow(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input));

    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    if (!checkBicubicBackwardUseFast(grad_output, align_corners, scales_h_attr, scales_w_attr, grad_input)) {
        return upsample_bicubic2d_aa_backward_out_slow(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
    }
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAAGrad,
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h_attr,
        scales_w_attr,
        grad_input);
    return grad_input;
}

at::Tensor _upsample_bicubic2d_aa_backward(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleBicubic2dAAGrad,
        upsample_bicubic2d_aa_backward_slow(grad_output, output_size, input_size, align_corners, scales_h, scales_w));

    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    if (!checkBicubicBackwardUseFast(grad_output, align_corners, scales_h_attr, scales_w_attr, grad_input)) {
        return upsample_bicubic2d_aa_backward_slow(
            grad_output, output_size, input_size, align_corners, scales_h, scales_w);
    }
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAAGrad,
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h_attr,
        scales_w_attr,
        grad_input);
    return grad_input;
}

}  // namespace op_api
