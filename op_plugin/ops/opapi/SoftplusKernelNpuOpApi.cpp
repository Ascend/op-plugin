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

at::Tensor& softplus_out(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSoftplus, acl_op::softplus_out(self, beta, threshold, result));
  npu_preparation::check_tensor({self}, result, self);
  EXEC_NPU_CMD(aclnnSoftplus, self, beta, threshold, result);
  return result;
}

at::Tensor softplus(
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold) {
  DO_COMPATIBILITY(aclnnSoftplus, acl_op::softplus(self, beta, threshold));
  auto output_size = op_infer::input_same_output_size(self);
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());
  op_api::softplus_out(self, beta, threshold, result);
  return result;
}
}
