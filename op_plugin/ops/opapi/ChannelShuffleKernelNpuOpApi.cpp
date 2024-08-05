// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor channel_shuffle(const at::Tensor& self, int64_t groups) {
  DO_COMPATIBILITY(aclnnChannelShuffle, acl_op::channel_shuffle(self, groups));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  EXEC_NPU_CMD(aclnnChannelShuffle, self, groups, result);
  return result;
}
}
