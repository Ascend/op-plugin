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
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/op_api_common.h"

#include <array>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using tensor_list = std::tuple<at::Tensor, at::Tensor>;

tensor_list _npu_moe_token_unpermute(
    const at::Tensor &permuted_tokens,
    const at::Tensor &sorted_indices,
    const c10::optional<at::Tensor> &probs,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape)
{
    auto unpermuted_tokens_size = op_infer::npu_moe_token_unpermute_out_size(permuted_tokens, sorted_indices, probs);
    at::Tensor unpermuted_tokens = npu_preparation::apply_tensor_without_format(
        unpermuted_tokens_size, permuted_tokens.options().dtype());

    std::array<int64_t, 1> default_restore_shape = {1};
    at::IntArrayRef restore_shape_value = restore_shape.value_or(default_restore_shape);
    EXEC_NPU_CMD(aclnnMoeTokenUnpermute,
        permuted_tokens,
        sorted_indices,
        probs,
        padded_mode,
        restore_shape_value,
        unpermuted_tokens);

    bool has_probs = probs.has_value() && probs.value().defined();
    bool need_save_permuted_tokens = has_probs || !op_plugin::utils::is_gte_cann_version_910();
    at::Tensor permuted_tokens_for_backward = need_save_permuted_tokens ? permuted_tokens : at::Tensor();
    return std::make_tuple(unpermuted_tokens, permuted_tokens_for_backward);
}

at::Tensor npu_moe_token_unpermute(
    const at::Tensor &permuted_tokens,
    const at::Tensor &sorted_indices,
    const c10::optional<at::Tensor> &probs,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape)
{
    tensor_list results = at_npu::native::custom_ops::_npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probs, padded_mode, restore_shape);
    return std::get<0>(results);
}
}  // namespace op_api
