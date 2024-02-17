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

namespace acl_op {
std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    at::Tensor& running_mean_opt,
    at::Tensor& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  return acl_op::native_batch_norm(
      self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train, momentum, eps);
}
} // namespace acl_op
