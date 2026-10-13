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

#include <c10/core/GradMode.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Workspace layout shared by the fused kernel and the composite fallback:
// 2D (B, 5H) planes [r | z | n | hx | gh_n]; r/z are the post-sigmoid gates,
// n is the post-tanh new gate, gh_n is the biased hidden new-gate input.
constexpr int64_t GRU_CELL_WORKSPACE_PLANES = 5;

// Same math as the aten reference (aten/src/ATen/native/RNN.cpp, GRUCell);
// also builds the (B, 5H) workspace for the backward when need_workspace is set.
std::tuple<at::Tensor, at::Tensor> thnn_fused_gru_cell_composite(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias,
    bool need_workspace)
{
    at::Tensor igates_b = (input_bias.has_value() && input_bias->defined())
        ? input_gates + input_bias.value()
        : input_gates;
    at::Tensor hgates_b = (hidden_bias.has_value() && hidden_bias->defined())
        ? hidden_gates + hidden_bias.value()
        : hidden_gates;

    auto chunked_igates = igates_b.unsafe_chunk(3, 1); // [reset, input, new]
    auto chunked_hgates = hgates_b.unsafe_chunk(3, 1);

    auto reset_gate = (chunked_igates[0] + chunked_hgates[0]).sigmoid_();
    auto input_gate = (chunked_igates[1] + chunked_hgates[1]).sigmoid_();
    auto new_gate = (chunked_igates[2] + reset_gate * chunked_hgates[2]).tanh_();

    auto hy = (hx - new_gate).mul_(input_gate).add_(new_gate);

    if (!need_workspace) {
        return std::make_tuple(hy, at::Tensor());
    }
    at::Tensor workspace = at::cat({reset_gate, input_gate, new_gate, hx, chunked_hgates[2]}, 1);
    return std::make_tuple(hy, workspace);
}

// The fused kernel requires the vendor package (detected by symbol lookup)
// and fp32/fp16/bf16 2D inputs.
bool thnn_fused_gru_cell_aclnn_supported(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx)
{
    static const bool is_aclnn_available = check_aclnn_kernel_available("aclnnThnnFusedGruCell");
    bool type_support = (input_gates.scalar_type() == at::kFloat) ||
        (input_gates.scalar_type() == at::kHalf) || (input_gates.scalar_type() == at::kBFloat16);
    bool rank_support = (input_gates.dim() == 2) && (hidden_gates.dim() == 2) && (hx.dim() == 2);
    return is_aclnn_available && type_support && rank_support;
}

// The workspace is only consumed by the backward, so build it only for calls
// that will be differentiated: grad mode on and some input requires grad.
bool thnn_fused_gru_cell_need_workspace(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias)
{
    return at::GradMode::is_enabled() &&
        (input_gates.requires_grad() || hidden_gates.requires_grad() || hx.requires_grad() ||
        (input_bias.has_value() && input_bias->defined() && input_bias->requires_grad()) ||
        (hidden_bias.has_value() && hidden_bias->defined() && hidden_bias->requires_grad()));
}
} // namespace

std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const c10::optional<at::Tensor>& input_bias,
    const c10::optional<at::Tensor>& hidden_bias)
{
    if (thnn_fused_gru_cell_aclnn_supported(input_gates, hidden_gates, hx)) {
        const int64_t batch_size = hx.size(0);
        const int64_t hidden_size = hx.size(1);
        at::Tensor hy = npu_preparation::apply_tensor_without_format(hx.sizes(), hx.options());
        at::Tensor workspace = npu_preparation::apply_tensor_without_format(
            {batch_size, GRU_CELL_WORKSPACE_PLANES * hidden_size}, hx.options());
        EXEC_NPU_CMD(
            aclnnThnnFusedGruCell, input_gates, hidden_gates, hx, input_bias, hidden_bias, hy, workspace);
        return std::make_tuple(std::move(hy), std::move(workspace));
    }
    bool need_workspace =
        thnn_fused_gru_cell_need_workspace(input_gates, hidden_gates, hx, input_bias, hidden_bias);
    return thnn_fused_gru_cell_composite(
        input_gates, hidden_gates, hx, input_bias, hidden_bias, need_workspace);
}
} // namespace op_api
