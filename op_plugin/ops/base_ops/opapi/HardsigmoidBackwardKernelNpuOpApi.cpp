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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor hardsigmoid_backward(const at::Tensor& grad_output, const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnHardsigmoidBackward, acl_op::hardsigmoid_backward(grad_output, self));
  at::ScalarType result_dtype = at::native::result_type(grad_output, self);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(result_dtype));
  EXEC_NPU_CMD(aclnnHardsigmoidBackward, grad_output, self, result);
  return result;
}

}  // namespace op_api
