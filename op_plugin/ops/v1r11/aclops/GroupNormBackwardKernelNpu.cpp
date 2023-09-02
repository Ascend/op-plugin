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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& dY,
    const at::Tensor& X,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& gamma_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, 3> grad_input_mask) {
  TORCH_NPU_WARN_ONCE("Warning: kernel [native_group_norm_backward] is not supported by NPU currently."
                      " Now this kernel is running on CPU.");
  at::Tensor dY_cpu = dY.to("cpu");
  at::Tensor X_cpu = X.to("cpu");
  at::Tensor mean_cpu = mean.to("cpu");
  at::Tensor rstd_cpu = rstd.to("cpu");
  const at::Tensor& gamma = c10::value_or_else(gamma_opt, [] {return at::Tensor();});
  at::Tensor gamma_opt_cpu = gamma.defined() ? gamma.to("cpu") : gamma;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> result = at::native_group_norm_backward(
      dY_cpu, X_cpu, mean_cpu, rstd_cpu, gamma_opt_cpu, N, C, HxW, group, grad_input_mask);
  at::Tensor dX = std::get<0>(result);
  at::Tensor dgamma = std::get<1>(result);
  at::Tensor dbeta = std::get<2>(result);

  dX = dX.defined() ? dX.to(X.device()) : dX;
  dgamma = dgamma.defined() ? dgamma.to(X.device()) : dgamma;
  dbeta = dbeta.defined() ? dbeta.to(X.device()) : dbeta;

  return std::make_tuple(dX, dgamma, dbeta);
}
} // namespace acl_op
