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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &_upsample_bicubic2d_aa_backward_out(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners,
                                                c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor &grad_input)
{
    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    at::Tensor self = grad_output;
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAAGrad, grad_output, output_size, input_size, align_corners, scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

at::Tensor _upsample_bicubic2d_aa_backward(const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners,
                                           c10::optional<double> scales_h, c10::optional<double> scales_w)
{
    at::Tensor self = grad_output;
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    EXEC_NPU_CMD(aclnnUpsampleBicubic2dAAGrad, grad_output, output_size, input_size, align_corners, scales_h_attr, scales_w_attr, grad_input);
    return grad_input;
}

} // namespace op_api
