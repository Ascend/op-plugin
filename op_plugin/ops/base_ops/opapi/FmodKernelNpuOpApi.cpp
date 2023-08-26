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

at::Tensor& fmod_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnFmodTensor, acl_op::fmod_out(self, other, result));
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  result.resize_(output_size);
  EXEC_NPU_CMD(aclnnFmodTensor, self, other, result);
  return result;
}

at::Tensor fmod(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnFmodTensor, acl_op::fmod(self, other));
  // calculate the output size and output dtype
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);

  // construct the output tensor of the NPU
  at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnFmodTensor, self, other, result);
  return result;
}

at::Tensor& fmod_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod_out(self, other, result));
  result.resize_(self.sizes());
  EXEC_NPU_CMD(aclnnFmodScalar, self, other, result);
  return result;
}

at::Tensor fmod(const at::Tensor &self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod(self, other));
  at::Tensor result = npu_preparation::apply_tensor_without_format(self);
  EXEC_NPU_CMD(aclnnFmodScalar, self, other, result);
  return result;
}

at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnInplaceFmodTensor, acl_op::fmod_(self, other));
  EXEC_NPU_CMD(aclnnInplaceFmodTensor, self, other);
  return self;
}

at::Tensor& fmod_(at::Tensor& self, const at::Scalar& other) {
  DO_COMPATIBILITY(aclnnInplaceFmodScalar, acl_op::fmod_(self, other));
  EXEC_NPU_CMD(aclnnInplaceFmodScalar, self, other);
  return self;
}
} // namespace op_api
