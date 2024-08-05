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

at::Tensor& threshold_out(
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Scalar& value,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnThreshold, acl_op::threshold_out(self, threshold, value, result));
  auto res_type = at::result_type(self, result);
  npu_preparation::check_tensor({self}, result, res_type, self.sizes());
  EXEC_NPU_CMD(aclnnThreshold, self, threshold, value, result);
  return result;
}

at::Tensor threshold(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnThreshold, acl_op::threshold(self, threshold, value));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  op_api::threshold_out(self, threshold, value, result);
  return result;
}

at::Tensor& threshold_(at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceThreshold, acl_op::threshold_(self, threshold, value));
  EXEC_NPU_CMD(aclnnInplaceThreshold, self, threshold, value);
  return self;
}
}
