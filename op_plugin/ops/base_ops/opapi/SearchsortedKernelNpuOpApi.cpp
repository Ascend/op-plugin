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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& searchsorted_out(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSearchSorted,
                   acl_op::searchsorted_out(sorted_sequence, self, out_int32,
                                            right, side_opt, sorter_opt, result));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  npu_preparation::check_tensor(
      {sorted_sequence, self},
      result,
      scalar_type,
      self.sizes());
  EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right, sorter_opt, result);
  return result;
}

at::Tensor searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Tensor& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  DO_COMPATIBILITY(aclnnSearchSorted,
                   acl_op::searchsorted(sorted_sequence, self, out_int32, right, side_opt, sorter_opt));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(scalar_type));
  EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right, sorter_opt, result);
  return result;
}

at::Tensor searchsorted(
    const at::Tensor& sorted_sequence,
    const at::Scalar& self,
    bool out_int32,
    bool right,
    const c10::optional<c10::string_view> side_opt,
    const c10::optional<at::Tensor>& sorter_opt) {
  DO_COMPATIBILITY(aclnnSearchSorteds,
                   acl_op::searchsorted(sorted_sequence, self, out_int32, right, side_opt, sorter_opt));
  at::ScalarType scalar_type = out_int32 ? at::kInt : at::kLong;
  at::Tensor result = npu_preparation::apply_tensor_without_format({}, sorted_sequence.options().dtype(scalar_type));
  EXEC_NPU_CMD(aclnnSearchSorteds, sorted_sequence, self, out_int32, right, sorter_opt, result);
  return result;
}
} // namespace op_api
