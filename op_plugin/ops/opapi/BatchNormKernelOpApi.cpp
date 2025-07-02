// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> _native_batch_norm_legit(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& running_mean,
    at::Tensor& running_var,
    bool training,
    double momentum,
    double eps)
{
    return at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}

at::Tensor batch_norm(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    bool cudnn_enabled)
{
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});
    const at::Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return at::Tensor();});
    const at::Tensor& running_var = c10::value_or_else(running_var_opt, [] {return at::Tensor();});
    return std::get<0>(at::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps));
}
}  // namespace op_api
