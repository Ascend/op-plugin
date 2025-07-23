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
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

tensor_list _npu_moe_token_unpermute_with_routing_map(
    const at::Tensor &permuted_tokens,
    const at::Tensor &sorted_indices,
    at::IntArrayRef restore_shape,
    const c10::optional<at::Tensor> &probs,
    const c10::optional<at::Tensor> &routing_map,
    bool drop_and_pad)
{
    at::Tensor unpermuted_tokens;
    at::Tensor out_index;
    at::Tensor permuted_token_id;
    at::Tensor permute_probs;

    unpermuted_tokens = npu_preparation::apply_tensor_without_format({restore_shape[0], restore_shape[1]}, permuted_tokens.options().dtype());
    if (probs.has_value()) {
        permute_probs = npu_preparation::apply_tensor_without_format(sorted_indices.sizes(), probs.value().options().dtype());
    }
    out_index = npu_preparation::apply_tensor_without_format(sorted_indices.sizes(), sorted_indices.options().dtype());
    permuted_token_id = npu_preparation::apply_tensor_without_format(sorted_indices.sizes(), sorted_indices.options().dtype());

    EXEC_NPU_CMD(aclnnMoeTokenUnpermuteWithRoutingMap,
        permuted_tokens,
        sorted_indices,
        routing_map,
        probs,
        drop_and_pad,
        restore_shape,
        unpermuted_tokens,
        out_index,
        permuted_token_id,
        permute_probs);
    return std::tie(unpermuted_tokens, out_index, permuted_token_id, permute_probs);
}

at::Tensor npu_moe_token_unpermute_with_routing_map(
    const at::Tensor &permuted_tokens,
    const at::Tensor &sorted_indices,
    at::IntArrayRef restore_shape,
    const c10::optional<at::Tensor> &probs,
    const c10::optional<at::Tensor> &routing_map,
    bool drop_and_pad)
{
    tensor_list results = at_npu::native::custom_ops::_npu_moe_token_unpermute_with_routing_map(permuted_tokens,
        sorted_indices, restore_shape, probs, routing_map, drop_and_pad);
    return std::get<0>(results);
}

}  // namespace op_api
