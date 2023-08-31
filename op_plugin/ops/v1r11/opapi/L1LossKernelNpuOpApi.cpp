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

at::Tensor& l1_loss_out(const at::Tensor& self,
                        const at::Tensor& target,
                        int64_t reduction,
                        at::Tensor& result) {
  DO_COMPATIBILITY(aclnnL1Loss, acl_op::l1_loss_out(self, target, reduction, result));
  // check if result on NPU
  TORCH_CHECK(torch_npu::utils::is_npu(result), "result with device ", result.device(),
              " doesn't match the desired device NPU");
  // 1. When reduction = 'none', shape of result must be the same as self.
  // 2. When reduction != 'none', result must be a 0-dimensional tensor.
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::broadcast_ops_npu_output_size(self, target);
  }
  // Shape of result must be the same as self, dtype has no limitation.
  if (result.sizes() != output_size) {
    result.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1Loss, self, target, reduction, result);
  return result;
}

at::Tensor l1_loss(const at::Tensor& self,
                   const at::Tensor& target,
                   int64_t reduction) {
  DO_COMPATIBILITY(aclnnL1Loss, acl_op::l1_loss(self, target, reduction));
  // construct the output tensor of NPU
  // 1. If reduction='none', the output size should be the same size as self.
  // 2. Otherwise pass {} to ApplyTensor.
  // 3. Dtype of output should be the same dtype as self.
  at::IntArrayRef output_size;
  if (reduction == at::Reduction::None) {
    output_size = op_infer::broadcast_ops_npu_output_size(self, target);
  }
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnL1Loss, self, target, reduction, result);
  return result;
}

} // namespace op_api
