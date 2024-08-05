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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& hardswish_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnHardswish, acl_op::hardswish_out(self, result));
  npu_preparation::check_tensor({self}, result, self);
  EXEC_NPU_CMD(aclnnHardswish, self, result);
  return result;
}

at::Tensor hardswish(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnHardswish, acl_op::hardswish(self));
  auto out_size = op_infer::input_same_output_size(self);
  auto result = npu_preparation::apply_tensor_without_format(out_size, self.options());
  EXEC_NPU_CMD(aclnnHardswish, self, result);
  return result;
}

at::Tensor& hardswish_(at::Tensor &self) {
  DO_COMPATIBILITY(aclnnInplaceHardswish, acl_op::hardswish_(self));
  EXEC_NPU_CMD(aclnnInplaceHardswish, self);
  return self;
}
}
