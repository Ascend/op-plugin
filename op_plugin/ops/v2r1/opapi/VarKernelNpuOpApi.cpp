// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var_out(self, dim, correction, keepdim, result));
  c10::SmallVector<int64_t, op_infer::SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value().toInt() : 1;

  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      self,
      output_size);

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarCorrection, self, rd, real_correction, keepdim, result);
  return result;
}

at::Tensor var(
    const at::Tensor & self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  DO_COMPATIBILITY(aclnnVarCorrection, acl_op::var(self, dim, correction, keepdim));
  c10::SmallVector<int64_t, op_infer::SIZE> real_dim = {};
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto real_correction = correction.has_value() ? correction.value().toInt() : 1;
  auto result = npu_preparation::apply_tensor_without_format(output_size, self.options());

  at_npu::native::OpPreparation::check_tensor(
      {self},
      result,
      self,
      output_size);

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarCorrection, self, rd, real_correction, keepdim, result);
  return result;
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> real_dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dim.has_value()) {
    real_dim = op_infer::array_to_small_vector(dim.value());
  }
  int64_t real_correction = correction.has_value() ? correction.value().toInt() : 1;
  auto output_size = op_infer::reduce_ops_npu_output_size(self, real_dim, keepdim);
  auto var = npu_preparation::apply_tensor_without_format(output_size, self.options());
  auto mean = npu_preparation::apply_tensor_without_format(output_size, self.options());

  auto rd = at::IntArrayRef(real_dim);
  EXEC_NPU_CMD(aclnnVarMean, self, rd, real_correction, keepdim, mean, var);
  return std::tuple<at::Tensor, at::Tensor>(mean, var);
}
} // namespace op_api
