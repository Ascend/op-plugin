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

at::Tensor& index_select_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIndexSelect, acl_op::index_select_out(self, dim, index, result));
  auto output_size = op_infer::index_select_npu_output_size(self, dim, index);
  npu_preparation::check_tensor({self, index}, result, self.scalar_type(), output_size);
  EXEC_NPU_CMD(aclnnIndexSelect, self, dim, index, result);
  return result;
}

at::Tensor index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  DO_COMPATIBILITY(aclnnIndexSelect, acl_op::index_select(self, dim, index));
  auto output_size = op_infer::index_select_npu_output_size(self, dim, index);
  at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
  EXEC_NPU_CMD(aclnnIndexSelect, self, dim, index, result);
  return result;
}

at::Tensor& index_select_out(const at::Tensor& self, at::Dimname dim, const at::Tensor& index, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnIndexSelect, acl_op::index_select_out(self, dim, index, result));
  return op_api::index_select_out(self, dimname_to_position(self, dim), index, result);
}

at::Tensor index_select(const at::Tensor& self, at::Dimname dim, const at::Tensor& index) {
  DO_COMPATIBILITY(aclnnIndexSelect, acl_op::index_select(self, dim, index));
  return op_api::index_select(self, dimname_to_position(self, dim), index);
}

}
