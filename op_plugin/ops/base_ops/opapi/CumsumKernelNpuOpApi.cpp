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

at::Tensor& cumsum_out(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCumsum, acl_op::cumsum_out(self, dim, dtype, result));
  npu_preparation::check_tensor({self}, result, self.sizes());

  aclDataType dtype_new = ACL_DT_UNDEFINED;
  if (!dtype.has_value()) {
    dtype_new = npu_preparation::convert_to_acl_data_type(result.scalar_type());
  } else {
    dtype_new = npu_preparation::convert_to_acl_data_type(dtype.value());
  }

  EXEC_NPU_CMD(aclnnCumsum, self, dim, dtype_new, result);
  at::namedinference::propagate_names(result, self);
  return result;
}

at::Tensor& cumsum_out(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype,
                       at::Tensor& result) {
  DO_COMPATIBILITY(aclnnCumsum, acl_op::cumsum_out(self, dim, dtype, result));
  return op_api::cumsum_out(self, dimname_to_position(self, dim), dtype, result);
}

at::Tensor cumsum(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  DO_COMPATIBILITY(aclnnCumsum, acl_op::cumsum(self, dim, dtype));

  at::Tensor result;
  aclDataType dtype_new = ACL_DT_UNDEFINED;
  if (dtype.has_value()) {
    result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(dtype.value()));
    dtype_new = npu_preparation::convert_to_acl_data_type(dtype.value());
  } else {
    if (self.scalar_type() == at::ScalarType::Bool) {
      result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kLong));
    } else {
      result = npu_preparation::apply_tensor_without_format(self);
    }
    dtype_new = npu_preparation::convert_to_acl_data_type(result.scalar_type());
  }

  EXEC_NPU_CMD(aclnnCumsum, self, dim, dtype_new, result);
  at::namedinference::propagate_names(result, self);
  return result;
}
} // namespace op_api
