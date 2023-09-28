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

at::Tensor& reciprocal_out(const at::Tensor& self, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnReciprocal, acl_op::reciprocal_out(self, result));

  auto output_size = op_infer::input_same_output_size(self);
  npu_preparation::check_tensor(
      {self},
      result,
      result.scalar_type(),
      output_size);

  EXEC_NPU_CMD(aclnnReciprocal, self, result);
  return result;
}

at::Tensor reciprocal(const at::Tensor& self)
{
  DO_COMPATIBILITY(aclnnReciprocal, acl_op::reciprocal(self));
  // calculate the output size
  auto output_size = op_infer::input_same_output_size(self);
  auto out_dtype = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_dtype));
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnReciprocal, self, result);
  return result;
}

at::Tensor& reciprocal_(at::Tensor& self)
{
  // DO_COMPATIBILITY(aclnnInplaceReciprocal, acl_op::reciprocal_(self));
  EXEC_NPU_CMD(aclnnInplaceReciprocal, self);
  return self;
}

}  // namespace op_api

