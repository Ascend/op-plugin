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

std::tuple<at::Tensor, at::Tensor> sort(const at::Tensor &self,
                                        c10::optional<bool> stable,
                                        int64_t dim,
                                        bool descending)
{
  DO_COMPATIBILITY(aclnnSort, acl_op::sort(self, stable, dim, descending));
  at::Tensor values = npu_preparation::apply_tensor_without_format(self);
  at::Tensor indices = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kLong));
  bool argStable = c10::value_or_else(stable, [] { return false; });
  EXEC_NPU_CMD(aclnnSort, self, argStable, dim, descending, values, indices);
  return std::tie(values, indices);
}

std::tuple<at::Tensor &, at::Tensor &> sort_out(const at::Tensor &self,
                                                c10::optional<bool> stable,
                                                int64_t dim,
                                                bool descending,
                                                at::Tensor &values,
                                                at::Tensor &indices)
{
  DO_COMPATIBILITY(aclnnSort, acl_op::sort_out(self, stable, dim, descending, values, indices));
  npu_preparation::check_tensor({self}, values, values.scalar_type(), self.sizes());
  npu_preparation::check_tensor({self}, indices, indices.scalar_type(), self.sizes());
  bool argStable = c10::value_or_else(stable, [] { return false; });
  EXEC_NPU_CMD(aclnnSort, self, argStable, dim, descending, values, indices);
  return std::tie(values, indices);
}
}  // namespace op_api

