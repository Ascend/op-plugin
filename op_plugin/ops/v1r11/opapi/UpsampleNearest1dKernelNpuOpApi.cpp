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

at::Tensor upsample_nearest1d(const at::Tensor& input,
                              c10::optional<at::IntArrayRef> output_size,
                              c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1d, op_api::upsample_nearest1d(input, output_size, scale_factors));
  auto compute_size = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
  auto scales_w = get_scale_value(scale_factors, 0);
  c10::SmallVector<int64_t, SIZE> out_size = upsample_linear1d_npu_output_size(input, compute_size, false, scales_w);
  at::Tensor result = npu_preparation::apply_tensor_without_format(input, out_size);
  auto result_size = at::IntArrayRef(compute_size);
  EXEC_NPU_CMD(aclnnUpsampleNearest1d, input, result_size, result);
  return result;
}
} // namespace op_api
