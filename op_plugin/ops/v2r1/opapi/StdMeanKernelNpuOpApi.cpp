// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

std::tuple<at::Tensor, at::Tensor> std_mean(
      const at::Tensor& self,
      at::DimnameList dim,
      const c10::optional<at::Scalar>& correction,
      bool keepdim) {
  return op_api::std_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}

std::tuple<at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<at::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnStdMeanCorrection, acl_op::std_mean(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, SIZE> real_dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);

  at::Tensor std_out = npu_preparation::apply_tensor_without_format(self, output_size);
  at::Tensor mean_out = npu_preparation::apply_tensor_without_format(self, output_size);

  int64_t real_correction = correction.has_value() ? correction.value().toInt() : 1;
  auto real_dim_array = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnStdMeanCorrection, self, real_dim_array, real_correction, keepdim, std_out, mean_out);
  return std::tie(std_out, mean_out);
}

std::tuple<at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return op_api::std_mean(self, dimnames_to_positions(self, dim), unbiased, keepdim);
}

std::tuple<at::Tensor, at::Tensor> std_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return op_api::std_mean(self, at::OptionalIntArrayRef(dim),
                          c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> std_mean(const at::Tensor& self, bool unbiased) {
  return op_api::std_mean(self, c10::nullopt, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), false);
}

} // namespace op_api