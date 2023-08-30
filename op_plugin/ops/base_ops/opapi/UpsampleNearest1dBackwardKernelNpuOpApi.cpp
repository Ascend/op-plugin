// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

namespace op_api{
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& upsample_nearest1d_backward_out(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales,
    at::Tensor& grad_input) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1dBackward,
                   acl_op::upsample_nearest1d_backward_out(grad_output, output_size, input_size, scales, grad_input));
  npu_preparation::check_tensor(
      {grad_output},
      grad_input,
      grad_output,
      input_size);
  double scales_attr = scales.value_or(-1);
  EXEC_NPU_CMD(aclnnUpsampleNearest1dBackward, grad_output, output_size, input_size, scales_attr, grad_input);
  return grad_input;
}

at::Tensor upsample_nearest1d_backward(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1dBackward,
                   acl_op::upsample_nearest1d_backward(grad_output, output_size, input_size, scales));
  at::Tensor grad_input = npu_preparation::apply_tensor_without_format(grad_output, input_size);
  double scales_attr = scales.value_or(-1);
  EXEC_NPU_CMD(aclnnUpsampleNearest1dBackward, grad_output, output_size, input_size, scales_attr,
               grad_input);
  return grad_input;
}  
  
} // namespace op_api

