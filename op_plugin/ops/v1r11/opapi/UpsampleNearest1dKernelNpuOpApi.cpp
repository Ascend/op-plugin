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

at::Tensor upsample_nearest1d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  DO_COMPATIBILITY(aclnnUpsampleNearest1d, acl_op::upsample_nearest1d(input, output_size, scale_factors));
  c10::SmallVector<int64_t, SIZE> out_size = op_infer::upsample_infershape_with_scale(input.sizes(), output_size,
                                                                                      scale_factors);
  at::Tensor result = npu_preparation::apply_tensor_without_format(input, out_size);
  
  EXEC_NPU_CMD(aclnnUpsampleNearest1d, input, output_size, result);
  return result;
}

} // namespace op_api