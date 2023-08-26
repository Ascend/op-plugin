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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
at::Tensor or___dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool is_self_wrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  if (is_self_wrapped) {
    return other;
  } else {
    return self;
  }
}

at::Tensor& or___out_scalar_npu(at::Tensor& result, const at::Tensor& self, at::Scalar other) {
  string real_op_name = (self.dtype() == at::ScalarType::Bool) ? "LogicalOr" : "BitwiseOr";
  at_npu::native::OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor& or___out_tensor_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    or___out_scalar_npu(result, self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    or___out_scalar_npu(result, other, self.item());
  } else {
    string real_op_name = (self.dtype() == at::kBool) ? "LogicalOr" : "BitwiseOr";
    at_npu::native::OpCommand cmd;
    cmd.Name(real_op_name)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}
} // namespace

at::Tensor __or__(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor output_tensor = or___dest_output(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::ApplyTensor(output_tensor, output_size);
  or___out_tensor_npu(result, self, other);
  return result;
}

at::Tensor __or__(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  or___out_scalar_npu(result, self, other);
  return result;
}
}  // namespace acl_op
