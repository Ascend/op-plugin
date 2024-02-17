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

// reduce value must be "add" or "multiply"
static inline bool reduce_valid(c10::string_view reduce)
{
  return (reduce == "add" || reduce == "multiply");
}

static int64_t get_reduce(c10::string_view reduce)
{
  if (reduce == "add") {
    return 1;
  } else if (reduce == "multiply") {
    return 2;
  }
  return 0;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnScatter, acl_op::scatter_out(self, dim, index, src, result));
  npu_preparation::check_tensor({self, src, index}, result, self);
  int64_t reduction = 0;
  EXEC_NPU_CMD(aclnnScatter, self, dim, index, src, reduction, result);
  return result;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Tensor& src, c10::string_view reduce, at::Tensor& result)
{
  npu_preparation::check_tensor({self, src, index}, result, self);
  TORCH_CHECK(reduce_valid(reduce), "Reduce should be either add or multiply");
  int64_t reduction = get_reduce(reduce);
  EXEC_NPU_CMD(aclnnScatter, self, dim, index, src, reduction, result);
  return result;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Scalar& value, at::Tensor& result)
{
  DO_COMPATIBILITY(aclnnScatterValue, acl_op::scatter_out(self, dim, index, value, result));
  npu_preparation::check_tensor({self, index}, result, self);
  int64_t reduction = 0;
  EXEC_NPU_CMD(aclnnScatterValue, self, dim, index, value, reduction, result);
  return result;
}

at::Tensor& scatter_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
    const at::Scalar& value, c10::string_view reduce, at::Tensor& result)
{
  npu_preparation::check_tensor({self, index}, result, self);
  TORCH_CHECK(reduce_valid(reduce), "Reduce should be either add or multiply");
  int64_t reduction = get_reduce(reduce);
  EXEC_NPU_CMD(aclnnScatterValue, self, dim, index, value, reduction, result);
  return result;
}
}  // namespace op_api

