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

at::Tensor binary_cross_entropy_with_logits(const at::Tensor& self, const at::Tensor& target,
                                            const c10::optional<at::Tensor>& weight_opt,
                                            const c10::optional<at::Tensor>& pos_weight_opt, int64_t reduction) {
  DO_COMPATIBILITY(aclnnBinaryCrossEntropyWithLogits,
                   acl_op::binary_cross_entropy_with_logits(self, target, weight_opt, pos_weight_opt, reduction));

  // calculate the output size
  at::IntArrayRef outputSize;
  if (reduction == at::Reduction::None) {
    outputSize = op_infer::input_same_output_size(target);
  } else {
    outputSize = at::ArrayRef<int64_t>();
  }

  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(target, outputSize);

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnBinaryCrossEntropyWithLogits, self, target, weight_opt, pos_weight_opt, reduction, result);
  return result;
}

}  // namespace op_api
