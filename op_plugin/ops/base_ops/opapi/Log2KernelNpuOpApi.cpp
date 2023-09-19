// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

at::Tensor& log2_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLog2, acl_op::log2_out(self, result));
  TORCH_CHECK(!isIntegralType(result.scalar_type(), true),
              "result type Float can't be cast to the desired output type ", toString(self.scalar_type()));
  at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnLog2, self, result);
  return result;
}

at::Tensor log2(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnLog2, acl_op::log2(self));
  // construct the output tensor of the NPU
  at::ScalarType out_dtype = self.scalar_type();
  if (isIntegralType(self.scalar_type(), true)) {
    out_dtype = at::kFloat;
  }
  at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(),
                                                                                 self.options().dtype(out_dtype));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLog2, self, result);

  return result;
}

at::Tensor& log2_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceLog2, acl_op::log2_(self));
  TORCH_CHECK(!isIntegralType(self.scalar_type(), true),
              "result type Float can't be cast to the desired output type ", toString(self.scalar_type()));
  EXEC_NPU_CMD(aclnnInplaceLog2, self);
  return self;
}
} // namespace op_api
