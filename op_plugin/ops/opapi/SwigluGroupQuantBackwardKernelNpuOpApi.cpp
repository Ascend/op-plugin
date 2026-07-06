// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor> npu_swiglu_group_quant_backward(const at::Tensor &grad_y, const at::Tensor &x,
    const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &y_origin,
    const c10::optional<at::Tensor> &group_index, double clamp_limit)
{

    // check x last dim
    int64_t x_last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(x_last_dim % 2 == 0, "x last dim size should be even", OPS_ERROR(ErrCode::PARAM));

    at::Tensor grad_x = npu_preparation::apply_tensor_without_format(x.sizes(), x.options());

    at::Tensor grad_weight;

    if (weight.has_value() && weight->defined()) {
        grad_weight  = npu_preparation::apply_tensor_without_format(weight.value().sizes(), weight.value().options());
    } else {
        grad_weight  = at::empty({0}, x.options().dtype(at::kFloat));
    }

    EXEC_NPU_CMD(aclnnSwigluGroupQuantGrad, grad_y, x, weight, y_origin, group_index, clamp_limit, grad_x, grad_weight);

    return std::make_tuple(grad_x, grad_weight);
}
} // namespace op_api
