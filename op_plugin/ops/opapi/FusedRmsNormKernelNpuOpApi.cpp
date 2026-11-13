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

// NOTE (rms_norm design v1.1): this shell calls the CANN official aclnnRmsNorm
// interface DIRECTLY (no reuse of the custom op op_api::npu_rms_norm) to keep
// the aten open-source path independent from the custom-op implementations, and
// aligns capability/edge behavior with _fused_rms_norm_cuda.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#if VERSION_BETWEEN(V2R9, VERSION_NEWEST)
#include <ATen/ops/_fused_rms_norm_compositeimplicitautograd_dispatch.h>
#include <limits>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> _fused_rms_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const c10::optional<at::Tensor>& weight_opt,
    c10::optional<double> eps) {
  TORCH_CHECK(!normalized_shape.empty() && normalized_shape.size() <= (size_t)input.dim(),
              "rms_norm(): normalized_shape must have 1 ~ input.dim() elements, but got ",
              normalized_shape.size());
  if (weight_opt.has_value() && weight_opt->defined()) {
    TORCH_CHECK(weight_opt->sizes().equals(normalized_shape),
                "Expected weight to be of same shape as normalized_shape, but got ",
                "weight of shape ", weight_opt->sizes(),
                " and normalized_shape = ", normalized_shape);
  }
  TORCH_CHECK(
      input.sizes().slice(input.dim() - normalized_shape.size()).equals(normalized_shape),
      "Given normalized_shape=", normalized_shape,
      ", expected input with shape [*, ", normalized_shape,
      "], but got input of size ", input.sizes());
  if (!is_ascend950_path() || input.scalar_type() == at::kDouble) {
    return at::compositeimplicitautograd::_fused_rms_norm(input, normalized_shape, weight_opt, eps);
  }
  auto acc_type = input.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  double eps_val;
  if (acc_type == at::ScalarType::Float) {
    eps_val = eps.value_or(std::numeric_limits<float>::epsilon());
  } else {
    eps_val = eps.value_or(std::numeric_limits<double>::epsilon());
  }
  at::Tensor weight = (weight_opt.has_value() && weight_opt->defined())
      ? *weight_opt : at::ones(normalized_shape, input.options());
  bool need_fold = (weight.dim() > 1) || (input.dim() == 1);
  at::Tensor x = need_fold ? input.reshape({-1, weight.numel()}) : input;
  at::Tensor gamma = weight.reshape({-1});
  const int64_t M = x.size(0), N = x.size(-1);
  x = *x.expect_contiguous();
  gamma = *gamma.expect_contiguous();
  const int64_t axis = input.dim() - (int64_t)normalized_shape.size();
  std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().begin() + axis);
  stat_shape.insert(stat_shape.end(), normalized_shape.size(), 1);
  std::vector<int64_t> y_sizes = need_fold ? std::vector<int64_t>{M, N} : input.sizes().vec();
  at::Tensor y = npu_preparation::apply_tensor_with_format(y_sizes, input.options(), ACL_FORMAT_ND);
  at::Tensor rstd = npu_preparation::apply_tensor_with_format(
      need_fold ? std::vector<int64_t>{M, 1} : stat_shape,
      input.options().dtype(at::kFloat), ACL_FORMAT_ND);

  if (x.numel() > 0) {
    EXEC_NPU_CMD(aclnnRmsNorm, x, gamma, eps_val, y, rstd);
  }
  if (need_fold) {
    y = y.reshape(input.sizes());
    rstd = rstd.view(stat_shape);
  }
  return std::make_tuple(y, rstd);
}

} // namespace op_api
#endif
