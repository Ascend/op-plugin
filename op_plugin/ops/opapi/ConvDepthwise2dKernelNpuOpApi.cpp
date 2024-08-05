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

const at::Tensor& _conv_depthwise2d_out(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    const at::Tensor& out) {
  DO_COMPATIBILITY(aclnnConvDepthwise2d, acl_op::_conv_depthwise2d_out(self, weight, kernel_size, bias_opt,
                                                                       stride, padding, dilation, out));
  if (!at_npu::native::env::CheckForbidInternalFormat() || !at_npu::native::env::CheckJitDisable()) {
    return acl_op::_conv_depthwise2d_out(self, weight, kernel_size, bias_opt, stride, padding, dilation, out);
  }
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
  EXEC_NPU_CMD(aclnnConvDepthwise2d, self, weight, kernel_size, bias, stride, padding, dilation, out, cube_math_type);
  return out;
}

at::Tensor _conv_depthwise2d(
    const at::Tensor& self,
    const at::Tensor& weight,
    at::IntArrayRef kernel_size,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation) {
  DO_COMPATIBILITY(aclnnConvDepthwise2d, acl_op::_conv_depthwise2d(self, weight, kernel_size, bias_opt,
                                                                   stride, padding, dilation));
  if (!at_npu::native::env::CheckForbidInternalFormat() || !at_npu::native::env::CheckJitDisable()) {
    return acl_op::_conv_depthwise2d(self, weight, kernel_size, bias_opt, stride, padding, dilation);
  }
  const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
  auto output_size = op_infer::conv_depthwise2d_npu_output_size(self, weight, kernel_size, stride, padding, dilation);
  at::Tensor out = npu_preparation::apply_tensor_without_format(self, output_size);
  int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
  EXEC_NPU_CMD(aclnnConvDepthwise2d, self, weight, kernel_size, bias, stride, padding, dilation, out, cube_math_type);
  return out;
}

}
