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

#include <ATen/WrapDimUtils.h>

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using calcu_op_util = at_npu::native::CalcuOpUtil;
using npu_utils = at_npu::native::NpuUtils;

namespace {
auto check_and_trans_dim(const at::Tensor& self, at::IntArrayRef dim) {
  std::vector<int64_t> result_dim;
  auto self_dim = self.dim();
  for (int64_t i = 0; i < dim.size(); i++) {
      int64_t tmp_dim = c10::maybe_wrap_dim(dim[i], self_dim);
      result_dim.emplace_back(tmp_dim);
  }
  std::sort(result_dim.begin(), result_dim.end());
  return result_dim;
}

auto get_result_names(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  auto names = self.names();
  std::vector<at::Dimname> result_names;
  for (int64_t i = 0; i < names.size(); i++) {
    result_names.emplace_back(names[i]);
  }
  if (!keepdim) {
    for (int64_t i = dim.size() - 1; i >= 0; i--) {
      int64_t need_remove_dim = dim[i];
      result_names.erase(result_names.begin() + need_remove_dim);
    }
  }
  return result_names;
}

at::Tensor& var_after_out_nocheck(
    at::Tensor& var,
    const at::Tensor& self,
    const at::Tensor& mean_broadcast,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  bool if_std = false;
  at_npu::native::OpCommand cmd;
  cmd.Name("ReduceStdV2Update")
      .Input(self)
      .Input(mean_broadcast)
      .Output(var)
      .Attr("dim", dim)
      .Attr("if_std", if_std)
      .Attr("unbiased", unbiased)
      .Attr("keepdim", keepdim)
      .Run();
  return var;
}

std::tuple<at::Tensor&, at::Tensor&> var_mean_compute(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  auto mean_output_size_keepdim = op_infer::var_npu_output_size(self, dim, true);
  auto mean_output_size_not_keepdim = op_infer::var_npu_output_size(self, dim, false);
  mean = at::mean(self, dim, false);
  mean.resize_(mean_output_size_keepdim);
  at::Tensor mean_broadcast = op_plugin::npu_broadcast(mean, self.sizes());
  if (!keepdim) {
    mean.resize_(mean_output_size_not_keepdim);
  }
  var_after_out_nocheck(variance, self, mean_broadcast, dim, unbiased, keepdim);
  return std::tuple<at::Tensor&, at::Tensor&>(variance, mean);
}

std::tuple<at::Tensor&, at::Tensor&> var_mean_out_nocheck(
    at::Tensor& variance,
    at::Tensor& mean,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  c10::SmallVector<int64_t, N> dim_now =
      dim.empty() ? calcu_op_util::GetDimlistForTensor(self) : c10::SmallVector<int64_t, N>(dim);
  auto ori_type = self.scalar_type();
  TORCH_CHECK((ori_type == c10::ScalarType::Half || ori_type == c10::ScalarType::Float),
      "Var Mean only support float16 or float32 type.");
  TORCH_CHECK((variance.scalar_type() == mean.scalar_type() && variance.scalar_type() == ori_type),
      "mean's type and variance' type must be equal to input's type.");
  var_mean_compute(variance, mean, self, dim_now, unbiased, keepdim);

  return std::tuple<at::Tensor&, at::Tensor&>(variance, mean);
}
} // namespace

at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  // check and trans dim
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = op_infer::var_npu_output_size(self, dim_now, keepdim);
  at::Tensor mean = npu_preparation::apply_tensor(self, output_size);

  npu_preparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    var_mean_out_nocheck(contiguous_result, mean, self, dim_now, unbiased, keepdim);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    var_mean_out_nocheck(result, mean, self, dim_now, unbiased, keepdim);
  }

  return result;
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dimnames_to_positions(self, dim), c10::make_optional<c10::Scalar>(unbiased ? 1 : 0),
      keepdim);
}

at::Tensor& var_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim,
    at::Tensor& result) {
  return at::var_out(result, self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor var(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = op_infer::var_npu_output_size(self, dim_now, keepdim);

  at::Tensor variance = npu_preparation::apply_tensor(self, output_size);
  at::Tensor mean = npu_preparation::apply_tensor(self, output_size);
  var_mean_out_nocheck(variance, mean, self, dim_now, unbiased, keepdim);
  return variance;
}

at::Tensor var(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim), correction, keepdim);
}

at::Tensor var(const at::Tensor& self, bool unbiased) {
  c10::SmallVector<int64_t, N> dim = calcu_op_util::GetDimlistForTensor(self);
  return at::var(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), false);
}

at::Tensor var(const at::Tensor& self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  return at::var(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

at::Tensor var(const at::Tensor& self, at::DimnameList dim, bool unbiased, bool keepdim) {
  return at::var(self, dimnames_to_positions(self, dim),
      c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  auto unbiased = !(correction.has_value() && correction.value().toInt() == 0);
  auto dim_now = check_and_trans_dim(self, dim.value_or(at::IntArrayRef{}));
  auto output_size = op_infer::var_npu_output_size(self, dim_now, keepdim);

  at::Tensor variance = npu_preparation::apply_tensor(self, output_size);
  at::Tensor mean = npu_preparation::apply_tensor(self, output_size);
  var_mean_out_nocheck(variance, mean, self, dim_now, unbiased, keepdim);

  return std::tuple<at::Tensor, at::Tensor>(variance, mean);
}

std::tuple<at::Tensor, at::Tensor> var_mean(const at::Tensor& self, bool unbiased) {
  c10::SmallVector<int64_t, SIZE> dim = calcu_op_util::GetDimlistForTensor(self);
  return at::var_mean(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), false);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dim, c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    bool unbiased,
    bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim),
      c10::make_optional<c10::Scalar>(unbiased ? 1 : 0), keepdim);
}

std::tuple<at::Tensor, at::Tensor> var_mean(
    const at::Tensor& self,
    at::DimnameList dim,
    const c10::optional<c10::Scalar>& correction,
    bool keepdim) {
  return at::var_mean(self, dimnames_to_positions(self, dim), correction, keepdim);
}
} // namespace op_plugin
