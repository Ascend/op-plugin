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

namespace op_api
{
    static int64_t parseActivationV2(const c10::optional<c10::string_view> &activation)
    {
        c10::string_view activation_str = activation.value_or("None");
        std::string input_activation = std::string(activation_str);
        if (input_activation == "silu") {
            return 1;
        } else if (input_activation == "swish") {
            return 2;
        }
        return 0;
    }

    static int64_t parseConvModeV2(const c10::optional<c10::string_view> &conv_mode)
    {
        c10::string_view conv_mode_str = conv_mode.value_or("default");
        std::string mode = std::string(conv_mode_str);
        if (mode == "pangu") {
            return 1;
        }
        return 0; // "default"
    }

    void npu_fused_causal_conv1d_v2(
        at::Tensor &x,
        const at::Tensor &weight,
        at::Tensor &conv_states,
        const c10::optional<at::Tensor> &query_start_loc,
        const c10::optional<at::Tensor> &cache_indices,
        const c10::optional<at::Tensor> &initial_state_mode,
        const c10::optional<at::Tensor> &bias,
        const c10::optional<at::Tensor> &num_accepted_tokens,
        c10::optional<c10::string_view> activation,
        c10::optional<int64_t> pad_slot_id,
        c10::optional<int64_t> run_mode,
        c10::optional<int64_t> residual_connection,
        c10::optional<int64_t> max_query_len,
        const c10::optional<at::Tensor> &num_computed_tokens,
        const c10::optional<at::Tensor> &block_idx_first_scheduled_token,
        const c10::optional<at::Tensor> &block_idx_last_scheduled_token,
        const c10::optional<at::Tensor> &initial_state_idx,
        c10::optional<int64_t> block_size,
        c10::optional<c10::string_view> conv_mode)
    {
        int64_t activation_value = parseActivationV2(activation);
        int64_t pad_slot_id_value = pad_slot_id.value_or(-1);
        int64_t run_mode_value = run_mode.value_or(0);
        int64_t max_query_len_value = max_query_len.value_or(-1);
        int64_t residual_connection_value = residual_connection.value_or(0);
        int64_t block_size_value = block_size.value_or(128);
        int64_t conv_mode_value = parseConvModeV2(conv_mode);

        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnInplaceFusedCausalConv1d,
            x, weight, conv_states,
            query_start_loc, cache_indices, initial_state_mode, bias,
            num_accepted_tokens, num_computed_tokens,
            block_idx_first_scheduled_token, block_idx_last_scheduled_token,
            initial_state_idx,
            activation_value, pad_slot_id_value, run_mode_value,
            max_query_len_value, residual_connection_value,
            block_size_value, conv_mode_value);
        return;
    }

} // namespace op_api
