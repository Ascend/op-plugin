// Copyright (c) 2024-2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &upsample_nearest_exact1d_backward_out_slow(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor &grad_input)
{
    at::Tensor grad_input_slow =
        at::_upsample_nearest_exact1d_backward(grad_output.cpu(), output_size, input_size, scales);
    grad_input.copy_(grad_input_slow);
    return grad_input;
}

at::Tensor upsample_nearest_exact1d_backward_slow(
    const at::Tensor &grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales)
{
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);

    at::Tensor grad_input_slow =
        at::_upsample_nearest_exact1d_backward(grad_output.cpu(), output_size, input_size, scales);
    grad_input.copy_(grad_input_slow);
    return grad_input;
}

at::Tensor &_upsample_nearest_exact1d_backward_out(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor &grad_input)
{
    DO_COMPATIBILITY(aclnnUpsampleNearestExact1dBackward,
        upsample_nearest_exact1d_backward_out_slow(grad_output, output_size, input_size, scales, grad_input));

    npu_preparation::check_tensor({grad_output}, grad_input, grad_output, input_size);
    at::Tensor self = grad_output;
    double scales_attr = scales.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleNearestExact1dBackward, grad_output, output_size, input_size, scales_attr, grad_input);
    return grad_input;
}

at::Tensor _upsample_nearest_exact1d_backward(const at::Tensor &grad_output, at::IntArrayRef output_size,
    at::IntArrayRef input_size, c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleNearestExact1dBackward,
        upsample_nearest_exact1d_backward_slow(grad_output, output_size, input_size, scales));

    at::Tensor self = grad_output;
    double scales_attr = scales.value_or(0);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
    EXEC_NPU_CMD(aclnnUpsampleNearestExact1dBackward, grad_output, output_size, input_size, scales_attr, grad_input);
    return grad_input;
}

}  // namespace op_api
