// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

at::Tensor& upsample_nearest3d_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnUpsampleNearest3d,
                   acl_op::upsample_nearest3d_out(input, output_size, scales_d, scales_h, scales_w, result));
  auto output_osize = op_infer::upsample_nearest3d_npu_output_size(input, output_size, scales_d, scales_h, scales_w);
  npu_preparation::check_tensor({input}, result, input, output_osize);
  double scales_d_attr = scales_d.value_or(0);
  double scales_h_attr = scales_h.value_or(0);
  double scales_w_attr = scales_w.value_or(0);
  EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_size, scales_d_attr, scales_h_attr, scales_w_attr, result);
  return result;
}

at::Tensor upsample_nearest3d(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  DO_COMPATIBILITY(aclnnUpsampleNearest3d,
                   acl_op::upsample_nearest3d(input, output_size, scales_d, scales_h, scales_w));
  auto output_osize = op_infer::upsample_nearest3d_npu_output_size(input, output_size, scales_d, scales_h, scales_w);
  at::Tensor result = npu_preparation::apply_tensor_without_format(input, output_osize);
  double scales_d_attr = scales_d.value_or(0);
  double scales_h_attr = scales_h.value_or(0);
  double scales_w_attr = scales_w.value_or(0);
  EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_size, scales_d_attr, scales_h_attr, scales_w_attr, result);
  return result;
}

} // namespace op_api
