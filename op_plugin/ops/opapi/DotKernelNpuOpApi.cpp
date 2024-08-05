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

at::Tensor& dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnDot, acl_op::dot_out(self, tensor, result));

  c10::SmallVector<int64_t, op_infer::N> output_size = {};
  npu_preparation::check_tensor({self, tensor}, result, output_size);

  EXEC_NPU_CMD(aclnnDot, self, tensor, result);
  return result;
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor) {
  DO_COMPATIBILITY(aclnnDot, acl_op::dot(self, tensor));

  c10::SmallVector<int64_t, op_infer::N> output_size = {};
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);

  EXEC_NPU_CMD(aclnnDot, self, tensor, result);
  return result;
}
}
