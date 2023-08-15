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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> batch_norm_backward_reduce_npu_impl(
    at::Tensor& sum_dy,
    at::Tensor& sum_dy_xmu,
    at::Tensor& grad_weight,
    at::Tensor& grad_bias,
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& weight,
    bool input_g,
    bool weight_g,
    bool bias_g,
    bool is_fully_fp16 = false) {
  at::Tensor sum_dy_sum;
  at::Tensor sum_dy_xmu_sum;
  at::Tensor grad_bias_sum;

  at::Tensor grad_out_scalar = grad_out.scalar_type() == at::kFloat ? grad_out :
      op_plugin::npu_dtype_cast(grad_out, at::kFloat);
  at::Tensor self_scalar = self.scalar_type() == at::kFloat ? self :
      op_plugin::npu_dtype_cast(self, at::kFloat);
  at::Tensor mean_scalar = mean.scalar_type() == at::kFloat ? mean :
      op_plugin::npu_dtype_cast(mean, at::kFloat);
  at::Tensor invstd_scalar = invstd.scalar_type() == at::kFloat ? invstd :
      op_plugin::npu_dtype_cast(invstd, at::kFloat);

  c10::SmallVector<int64_t, N> axes;
  int dimN = self_scalar.ndimension();
  for (int i = 0; i < dimN; i++) {
    if (i == 1) {
      continue;
    }
    axes.emplace_back(i);
  }

  at::Tensor mul_dy_dx = grad_out_scalar * self_scalar;
  sum_dy_xmu_sum = at::sum(mul_dy_dx, axes, false);
  grad_bias_sum = at::sum(grad_out_scalar, axes, false);
  sum_dy_sum = grad_bias_sum;

  at::Tensor sum_dy_xmu_out = npu_preparation::ApplyTensor(sum_dy_sum);
  at::Tensor grad_weight_res = npu_preparation::ApplyTensor(invstd_scalar);

  at_npu::native::OpCommand cmd;
  cmd.Name("SyncBatchNormBackwardReduce")
      .Input(sum_dy_sum)
      .Input(sum_dy_xmu_sum)
      .Input(mean_scalar)
      .Input(invstd_scalar)
      .Output(sum_dy_xmu_out)
      .Output(grad_weight_res)
      .Run();

  if (input_g) {
    sum_dy_xmu.copy_(sum_dy_xmu_out);
    sum_dy.copy_(sum_dy_sum);
  }
  if (weight_g) {
    grad_weight.copy_(grad_weight_res);
  }
  if (bias_g) {
    grad_bias.copy_(grad_bias_sum);
  }
  return std::tie(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_backward_reduce(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  TORCH_CHECK(self.scalar_type() == grad_out.scalar_type(),
      "Expected input's dtype equal grad_out's dtype ",
      grad_out.scalar_type(), "But found ", self.scalar_type());

  bool is_fully_fp16 = false;
  if (self.scalar_type() == mean.scalar_type() && self.scalar_type() == at::kHalf) {
    is_fully_fp16 = true;
  }

  int64_t n_input = self.size(1);
  at::Tensor sum_dy_val;
  at::Tensor sum_dy_xmu_val;
  at::Tensor grad_weight_val;
  at::Tensor grad_bias_val;

  auto fp_type = is_fully_fp16 ? at::kHalf : at::kFloat;
  at::Tensor weight_val = weight.defined() ? weight : at::ones({n_input}, self.options().dtype(fp_type));

  auto mean_dtype = mean.options().dtype(fp_type);
  if (input_g) {
      sum_dy_val = npu_preparation::ApplyTensor(mean, mean_dtype);
      sum_dy_xmu_val = npu_preparation::ApplyTensor(mean, mean_dtype);
  }

  auto weight_dtype = weight_val.options().dtype(fp_type);
  if (weight_g) {
      grad_weight_val = npu_preparation::ApplyTensor({n_input}, weight_dtype, weight_val);
  }
  if (bias_g) {
      grad_bias_val = npu_preparation::ApplyTensor({n_input}, weight_dtype, weight_val);
  }

  batch_norm_backward_reduce_npu_impl(sum_dy_val, sum_dy_xmu_val, grad_weight_val, grad_bias_val,
      grad_out, self, mean, invstd, weight, input_g, weight_g, bias_g, is_fully_fp16);
  return std::tie(sum_dy_val, sum_dy_xmu_val, grad_weight_val, grad_bias_val);
}
} // namespace op_plugin