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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;

namespace {
at::Tensor and_dest_output(const at::Tensor& self, const at::Tensor& other) {
  bool isSelfWrapped = calcu_op_util::IsScalarWrappedToTensor(self);
  if (not isSelfWrapped) {
    return self;
  } else {
    return other;
  }
}

at::Tensor& and_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar other) {
  at_npu::native::OpCommand cmd;
  cmd.Name((self.scalar_type() == at::kBool) ? "LogicalAnd" : "BitwiseAnd")
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();
  return result;
}

at::Tensor& and_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other) {
  if (npu_preparation::IsCPUScalar(other)) {
    and_out_npu_nocheck(result, self, other.item());
  } else if (npu_preparation::IsCPUScalar(self)) {
    and_out_npu_nocheck(result, other, self.item());
  } else {
    at_npu::native::OpCommand cmd;
    cmd.Name((self.scalar_type() == at::ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }
  return result;
}
} // namespace

at::Tensor __and__(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor outputTensor = and_dest_output(self, other);
  auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
  at::Tensor result = npu_preparation::ApplyTensor(outputTensor, output_size);
  and_out_npu_nocheck(result, self, other);
  return result;
}

at::Tensor __and__(const at::Tensor& self, const at::Scalar& other) {
  at::Tensor result = npu_preparation::ApplyTensor(self);
  and_out_npu_nocheck(result, self, other);
  return result;
}
}  // namespace acl_op
