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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::SmallVector<int64_t, SIZE> where_npu_output_size(const at::Tensor& condition) {
  int64_t dim = condition.dim();
  at::Tensor boolSelf = at_npu::native::custom_ops::npu_dtype_cast(condition, at::ScalarType::Bool);
  at::Tensor intSelf = at_npu::native::custom_ops::npu_dtype_cast(boolSelf, at::ScalarType::Int);
  at::Tensor cout_nonzero_self = at::sum(intSelf, at::ScalarType::Int);
  int64_t nonzero_num = cout_nonzero_self.item().toInt();
  at::SmallVector<int64_t, SIZE> output_size = {nonzero_num, dim};
  return output_size;
}
} // namespace

std::vector<at::Tensor> where(const at::Tensor& condition) {
  at::Tensor format_cast_of_condition = condition;
  if (npu_preparation::get_tensor_npu_format(condition) != ACL_FORMAT_ND) {
    format_cast_of_condition =
        at_npu::native::custom_ops::npu_format_cast(format_cast_of_condition, ACL_FORMAT_ND);
  }
  if (condition.scalar_type() == at::ScalarType::Half) {
    format_cast_of_condition = at_npu::native::custom_ops::npu_dtype_cast(format_cast_of_condition, at::ScalarType::Float);
  }

  auto output_size = where_npu_output_size(format_cast_of_condition);
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      output_size, format_cast_of_condition.options().dtype(at::kLong), ACL_FORMAT_ND);

  at_npu::native::OpCommand cmd;
  cmd.Name("NonZero")
      .Input(format_cast_of_condition)
      .Output(result)
      .Run();
  result = result.transpose(1, 0);
  std::vector<at::Tensor> chunk_result = result.chunk(result.size(0), 0);
  std::vector<at::Tensor> squeeze_result;
  for (uint64_t i = 0; i < chunk_result.size(); i++) {
    squeeze_result.push_back(chunk_result[i].squeeze(0));
  }

  return squeeze_result;
}
} // namespace acl_op
