// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor> npu_moe_token_unpermute_with_routing_map_grad(
    const at::Tensor &unpermuted_tokens_grad,
    const at::Tensor &out_index,
    const at::Tensor &permuted_token_id,
    const c10::optional<at::Tensor> &routing_map,
    const c10::optional<at::Tensor> &permuted_tokens,
    const c10::optional<at::Tensor> &probs,
    bool drop_and_pad,
    at::IntArrayRef restore_shape)
{
    at::Tensor permuted_tokens_grad_out;
    at::Tensor probs_grad_out;

    permuted_tokens_grad_out = npu_preparation::apply_tensor_without_format(
        {out_index.sizes()[0], unpermuted_tokens_grad.sizes()[1]}, unpermuted_tokens_grad.options().dtype());
    if (probs.has_value()) {
        probs_grad_out = npu_preparation::apply_tensor_without_format(
            probs.value().sizes(), unpermuted_tokens_grad.options().dtype());
    }

    EXEC_NPU_CMD(aclnnMoeTokenUnpermuteWithRoutingMapGrad,
        unpermuted_tokens_grad,
        out_index,
        permuted_token_id,
        routing_map,
        permuted_tokens,
        probs,
        drop_and_pad,
        restore_shape,
        permuted_tokens_grad_out,
        probs_grad_out);
    return std::tie(permuted_tokens_grad_out, probs_grad_out);
}
}  // namespace op_api
