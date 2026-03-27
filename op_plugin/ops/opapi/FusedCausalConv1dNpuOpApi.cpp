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
    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor> npu_fused_causal_conv1d_functional(
        const at::Tensor &x,
        const at::Tensor &weight,
        const at::Tensor &conv_states,
        const c10::optional<at::Tensor> &query_start_loc,
        const c10::optional<at::Tensor> &cache_indices,
        const c10::optional<at::Tensor> &initial_state_mode,
        const c10::optional<at::Tensor> &bias,
        const c10::optional<at::Tensor> &num_accepted_tokens,
        c10::optional<c10::string_view> activation_mode,
        c10::optional<int64_t> pad_slot_id,
        c10::optional<int64_t> run_mode,
        c10::optional<int64_t> residual_connection)
    {
        auto output_size = x.sizes();
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size, x.options());
        auto conv_states_clone = conv_states.clone(at::MemoryFormat::Contiguous);

        int64_t run_mode_value = run_mode.value_or(0);
        int64_t pad_slot_id_value = pad_slot_id.value_or(-1);
        int64_t residual_connection_value = residual_connection.value_or(0);
        c10::string_view activation_str = activation_mode.value_or("None");
        std::string input_activation = std::string(activation_str);
        int64_t activation_value = 0;
        if (input_activation == "silu")
        {
            activation_value = 1;
        }
        else if (input_activation == "swish")
        {
            activation_value = 2;
        }
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedCausalConv1d, x, weight, conv_states_clone,
                                     query_start_loc, cache_indices, initial_state_mode, bias, num_accepted_tokens,
                                     activation_value, pad_slot_id_value, run_mode_value, residual_connection_value, y);
        return std::make_tuple(y, conv_states_clone);
    }

    at::Tensor npu_fused_causal_conv1d(
        const at::Tensor &x,
        const at::Tensor &weight,
        at::Tensor &conv_states,
        const c10::optional<at::Tensor> &query_start_loc,
        const c10::optional<at::Tensor> &cache_indices,
        const c10::optional<at::Tensor> &initial_state_mode,
        const c10::optional<at::Tensor> &bias,
        const c10::optional<at::Tensor> &num_accepted_tokens,
        c10::optional<c10::string_view> activation_mode,
        c10::optional<int64_t> pad_slot_id,
        c10::optional<int64_t> run_mode,
        c10::optional<int64_t> residual_connection)
    {
        auto output_size = x.sizes();
        at::Tensor y = npu_preparation::apply_tensor_without_format(output_size, x.options());

        int64_t run_mode_value = run_mode.value_or(0);
        int64_t pad_slot_id_value = pad_slot_id.value_or(-1);
        int64_t residual_connection_value = residual_connection.value_or(0);
        c10::string_view activation_str = activation_mode.value_or("None");
        std::string input_activation = std::string(activation_str);
        int64_t activation_value = 0;
        if (input_activation == "silu")
        {
            activation_value = 1;
        }
        else if (input_activation == "swish")
        {
            activation_value = 2;
        }
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedCausalConv1d, x, weight, conv_states,
                                     query_start_loc, cache_indices, initial_state_mode, bias, num_accepted_tokens,
                                     activation_value, pad_slot_id_value, run_mode_value, residual_connection_value, y);
        return y;
    }

}