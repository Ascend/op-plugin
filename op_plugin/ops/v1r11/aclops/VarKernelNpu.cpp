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
at::Tensor& var_out(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim,
    at::Tensor& result) {
  c10::SmallVector<int64_t, N> dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dims.has_value()) {
    dim = op_infer::array_to_small_vector(dims.value());
  }
  bool unbiased = !(correction.has_value() && correction.value() == 0);
  int64_t real_correction = correction.has_value() ? correction.value() : 1;
  return cal_var_out(self, dim, real_correction, unbiased, keepdim, result);
}

at::Tensor var(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dims.has_value()) {
    dim = op_infer::array_to_small_vector(dims.value());
  }
  int64_t real_correction = correction.has_value() ? correction.value() : 1;
  bool unbiased = !(correction.has_value() && correction.value() == 0);
  return cal_var(self, dim, real_correction, unbiased, keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    c10::optional<at::IntArrayRef> dims,
    c10::optional<int64_t> correction,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim = op_plugin::utils::get_dimlist_for_tensor(self);
  if (dims.has_value()) {
    dim = op_infer::array_to_small_vector(dims.value());
  }
  bool unbiased = !(correction.has_value() && correction.value() == 0);
  int64_t real_correction = correction.has_value() ? correction.value() : 1;
  return cal_var_mean(self, dim, unbiased, real_correction, keepdim);
}
} // namespace acl_op
