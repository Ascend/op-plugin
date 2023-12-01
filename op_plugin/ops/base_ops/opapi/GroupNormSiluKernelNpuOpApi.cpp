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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_group_norm_silu(
    const at::Tensor& X,
    const c10::optional<at::Tensor>& gamma_opt,
    const c10::optional<at::Tensor>& beta_opt,
    int64_t group,
    double eps)

{
    at::Tensor y = npu_preparation::apply_tensor_without_format(X);
    auto x_size = op_infer::array_to_small_vector(X.sizes());
    at::Tensor mean = npu_preparation::apply_tensor_without_format(X, {x_size[0], group});
    at::Tensor rstd = npu_preparation::apply_tensor_without_format(X, {x_size[0], group});

    EXEC_NPU_CMD(aclnnGroupNormSilu, X, gamma_opt, beta_opt, group, eps, y, mean, rstd);
    return std::make_tuple(y, mean, rstd);
}

} // namespace op_api
