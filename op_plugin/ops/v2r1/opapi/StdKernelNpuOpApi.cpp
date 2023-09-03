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
#include "op_plugin/utils/custom_functions/opapi/inner_compute_op_api.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std_out(self, dim, unbiased, keepdim, result));
  return op_api::std_out(self, at::OptionalIntArrayRef(dim), int64_t{unbiased ? 1 : 0}, keepdim, result);
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std_out(self, dim, unbiased, keepdim, result));
  return op_api::std_out(self, dimnames_to_positions(self, dim), unbiased, keepdim, result);
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std_out(self, dim, correction, keepdim, result));
  return op_api::std_out(self, dimnames_to_positions(self, dim), correction, keepdim, result);
}

at::Tensor& std_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std_out(self, dim, correction, keepdim, result));
  c10::SmallVector<int64_t, SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value().toInt() : 1;
  npu_preparation::check_tensor({self}, result, self, output_size);
  EXEC_NPU_CMD(aclnnStd, self, dim.value(), real_correction, keepdim, result);
  return result;
}

at::Tensor std(const at::Tensor& self, bool unbiased) {
  return at::std(self, c10::nullopt, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0));
}

at::Tensor std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::std(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor std(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::std(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

at::Tensor std(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std(self, dim, correction, keepdim));
  return op_api::std(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor std(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStd, acl_op::std(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto result = npu_preparation::apply_tensor_without_format(output_size, self.options());
  return op_api::std_out(self, dim, correction, keepdim, result);
}


} // namespace op_api
