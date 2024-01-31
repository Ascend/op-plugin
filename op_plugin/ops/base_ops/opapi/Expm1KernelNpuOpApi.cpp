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

at::Tensor& expm1_out(const at::Tensor& self, at::Tensor& out) {
  DO_COMPATIBILITY(aclnnExpm1, acl_op::expm1_out(self, out));
  // resize_ the output size when size of out and self don't match with each other.
  if (out.sizes() != self.sizes()) {
    auto output_size = op_infer::input_same_output_size(self);
    out.resize_(output_size);
  }
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnExpm1, self, out);
  return out;
}

at::Tensor expm1(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnExpm1, acl_op::expm1(self));
  // construct the output tensor of NPU. If dtype of self isn't included in floating point list,
  // dtype of out must be float32.
  auto output_size = op_infer::input_same_output_size(self);
  at::ScalarType out_type = self.scalar_type();
  if (!isFloatingType(self.scalar_type())) {
    out_type = at::kFloat;
  }
  at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_type));
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnExpm1, self, out);
  return out;
}

at::Tensor& expm1_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceExpm1, acl_op::expm1_(self));
  // dispatch hostAPI
  EXEC_NPU_CMD(aclnnInplaceExpm1, self);
  return self;
}

} // namespace op_api
