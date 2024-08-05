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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>npu_add_layer_norm_backward(const c10::optional<at::Tensor> &dy_opt,
                                                                                      const at::Tensor &x1,
                                                                                      const at::Tensor &x2,
                                                                                      const at::Tensor &rstd,
                                                                                      const at::Tensor &mean,
                                                                                      const at::Tensor &gamma,
                                                                                      const c10::optional<at::Tensor> &dsum_opt)
{
    DO_COMPATIBILITY(aclnnAddLayerNormGrad, acl_op::npu_add_layer_norm_backward(dy_opt, x1, x2, rstd, mean, gamma, dsum_opt));
    const at::Tensor& dy = c10::value_or_else(dy_opt, [] {return at::Tensor();});
    const at::Tensor& dsum = c10::value_or_else(dsum_opt, [] {return at::Tensor();});
    auto d_y = dy.defined() ? dy : at::zeros(x1.sizes(), x1.options());
    auto d_sum = dsum.defined() ? dsum : at::zeros(x1.sizes(), x1.options());

    at::SmallVector<int64_t, SIZE> shape;
    for (int64_t index = 0; index < gamma.dim(); index++) {
        shape.emplace_back(gamma.size(index));
    }

    at::Tensor dx = npu_preparation::apply_tensor(x1);
    at::Tensor dgamma = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor dbeta = npu_preparation::apply_tensor(shape, x1.options().dtype(at::kFloat), x1);
    EXEC_NPU_CMD(aclnnAddLayerNormGrad, d_y, x1, x2, rstd, mean, gamma, d_sum, dx, dgamma, dbeta);
    return std::make_tuple(dx, dx, dgamma, dbeta);
}
}
