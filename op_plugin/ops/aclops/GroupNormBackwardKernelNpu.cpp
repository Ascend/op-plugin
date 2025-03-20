// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
namespace {
const int DIM = 3;
const int LAST_AXIS = 2;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const c10::optional<at::Tensor>& weight,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    std::array<bool, DIM> output_mask)
{
    TORCH_CHECK(group != 0, "group = 0 for group_norm_backward, please check!"
        + OPS_ERROR(ErrCode::VALUE));
    at::Tensor dY_reshaped_3;
    at::Tensor X_reshaped_3;
    if (input.dim() != DIM) {
        dY_reshaped_3 = grad_out.view({N, C, HxW});
        X_reshaped_3 = input.view({N, C, HxW});
    } else {
        dY_reshaped_3 = grad_out;
        X_reshaped_3 = input;
    }

    const at::Tensor& gamma = c10::value_or_else(weight, [] {return at::Tensor();});
    at::Tensor dY_b = gamma.defined() ? dY_reshaped_3.mul(gamma.view({1, C, 1})) : dY_reshaped_3;

    at::Tensor X_reshaped = X_reshaped_3.view({1, N * group, N > 0 ? -1 : 1});
    at::Tensor dY_reshaped = dY_b.view({1, N * group, N > 0 ? -1 : 1});
    at::Tensor mean_reshaped = mean.view({N * group});
    double eps = 1e-5;
    at::Tensor variance = 1.0 / rstd.mul(rstd) - eps;
    at::Tensor rstd_reshaped = variance.view({N * group});
    at::Tensor weight_opt = at::ones({N * group}, X_reshaped_3.options());

    auto output = at::native_batch_norm_backward(dY_reshaped, X_reshaped, weight_opt, c10::nullopt, c10::nullopt,
                                                 mean_reshaped, rstd_reshaped, true, eps, output_mask);

    at::Tensor dX = std::get<0>(output);
    dX = dX.view(X_reshaped_3.sizes());
    dX = dX.view(input.sizes());

    at::Tensor dgamma;
    at::Tensor dbeta;
    if (output_mask[1]) {
        at::Tensor mean_broadcast;
        at::Tensor rstd_broadcast;
        if (mean.sizes().size() == 1) {
            mean_broadcast = mean.view({N, group}).unsqueeze(LAST_AXIS);
            rstd_broadcast = rstd.view({N, group}).unsqueeze(LAST_AXIS);
        } else {
            mean_broadcast = mean.unsqueeze(LAST_AXIS);
            rstd_broadcast = rstd.unsqueeze(LAST_AXIS);
        }

        mean_broadcast = mean_broadcast.expand({N, group, C / group});
        mean_broadcast = mean_broadcast.reshape({N, C, 1});
        rstd_broadcast = rstd_broadcast.expand({N, group, C / group});
        rstd_broadcast = rstd_broadcast.reshape({N, C, 1});
        at::Tensor x_hat = at::sub(X_reshaped_3, mean_broadcast).mul(rstd_broadcast);
        dgamma = at::sum(at::sum(dY_reshaped_3.mul(x_hat), LAST_AXIS), 0);
    }

    if (output_mask[LAST_AXIS]) {
        dbeta = at::sum(at::sum(dY_reshaped_3, LAST_AXIS), 0);
    }
    return std::make_tuple(dX, dgamma, dbeta);
}

} // namespace acl_op
