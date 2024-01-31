// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either Eluress or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& elu_out(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                    const at::Scalar& input_scale, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnElu, acl_op::elu_out(self, alpha, scale, input_scale, result));
  npu_preparation::check_tensor({self}, result, self.sizes());
  EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
               const at::Scalar& input_scale) {
  DO_COMPATIBILITY(aclnnElu, acl_op::elu(self, alpha, scale, input_scale));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, result);
  return result;
}

at::Tensor& elu_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale) {
  DO_COMPATIBILITY(aclnnInplaceElu, acl_op::elu_(self, alpha, scale, input_scale));
  EXEC_NPU_CMD(aclnnInplaceElu, self, alpha, scale, input_scale);
  return self;
}
} // namespace op_api
