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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

namespace {
constexpr int64_t GRU_CELL_WORKSPACE_PLANES = 5;

// Backward decomposition over the workspace planes [r | z | n | hx | gh_n],
// used when the fused kernel is unavailable. fp16/bf16 inputs are computed in
// fp32 and cast back so the batch reduction for the bias gradients stays
// accurate.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
thnn_fused_gru_cell_backward_composite(const at::Tensor& grad_hy, const at::Tensor& workspace, bool has_bias)
{
    const auto in_dtype = grad_hy.scalar_type();
    const bool use_float_acc = (in_dtype == at::kHalf || in_dtype == at::kBFloat16);
    at::Tensor go = grad_hy;
    at::Tensor ws = workspace;
    if (use_float_acc) {
        go = grad_hy.to(at::kFloat);
        ws = workspace.to(at::kFloat);
    }

    const int64_t hidden_size = go.size(1);
    auto r = ws.narrow(1, 0, hidden_size);
    auto z = ws.narrow(1, hidden_size, hidden_size);
    auto n = ws.narrow(1, 2 * hidden_size, hidden_size);
    auto hx = ws.narrow(1, 3 * hidden_size, hidden_size);
    auto gh_n = ws.narrow(1, 4 * hidden_size, hidden_size);

    auto gin = go * (1 - z) * (1 - n * n);   // grad wrt new-gate pre-tanh
    auto gig = go * (hx - n) * (1 - z) * z;  // grad wrt update-gate pre-sigmoid
    auto grg = gin * gh_n * (1 - r) * r;     // grad wrt reset-gate pre-sigmoid
    auto ghn = gin * r;                      // grad wrt hidden new-gate
    auto ghx = go * z;                       // grad wrt hx

    at::Tensor grad_input_gates = at::cat({grg, gig, gin}, 1);
    at::Tensor grad_hidden_gates = at::cat({grg, gig, ghn}, 1);
    // has_bias=false: bias outputs are undefined (matches aten CUDA RNN.cu:661-663).
    at::Tensor grad_input_bias = has_bias ? grad_input_gates.sum(0) : at::Tensor();
    at::Tensor grad_hidden_bias = has_bias ? grad_hidden_gates.sum(0) : at::Tensor();

    if (use_float_acc) {
        grad_input_gates = grad_input_gates.to(in_dtype);
        grad_hidden_gates = grad_hidden_gates.to(in_dtype);
        ghx = ghx.to(in_dtype);
        if (has_bias) {
            grad_input_bias = grad_input_bias.to(in_dtype);
            grad_hidden_bias = grad_hidden_bias.to(in_dtype);
        }
    }
    return std::make_tuple(std::move(grad_input_gates), std::move(grad_hidden_gates), std::move(ghx),
        std::move(grad_input_bias), std::move(grad_hidden_bias));
}
} // namespace

// The backward math needs the gate values saved by the forward; same checks
// as aten checkGRUBackwardSizes (RNN.cu:628). The rank check runs first
// because the size check below indexes size(1).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _thnn_fused_gru_cell_backward(
    const at::Tensor& grad_hy,
    const at::Tensor& workspace,
    bool has_bias)
{
    TORCH_CHECK(workspace.defined(),
        "workspace is undefined: fused GRU backward requires the workspace saved by the forward",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grad_hy.dim() == 2 && workspace.dim() == 2,
        "expected grad_hy and workspace to be 2D, but got ", grad_hy.dim(), " and ", workspace.dim(),
        " dims respectively", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(workspace.size(0) == grad_hy.size(0) &&
        workspace.size(1) == grad_hy.size(1) * GRU_CELL_WORKSPACE_PLANES,
        "expected workspace to be 2D with size (B, 5H) = (", grad_hy.size(0), ", ",
        grad_hy.size(1) * GRU_CELL_WORKSPACE_PLANES, ") to match grad_hy, but got ", workspace.sizes(),
        OPS_ERROR(ErrCode::PARAM));

    static const bool fused_available = check_aclnn_kernel_available("aclnnThnnFusedGruCellBackward");
    if (fused_available) {
        // grad_hy/workspace may be non-contiguous: the aclnn descriptors carry strides.
        const auto batch_size = workspace.size(0);
        const auto hidden_size = workspace.size(1) / GRU_CELL_WORKSPACE_PLANES;

        at::Tensor grad_input_gates = at::empty({batch_size, 3 * hidden_size}, workspace.options());
        at::Tensor grad_hidden_gates = at::empty({batch_size, 3 * hidden_size}, workspace.options());
        at::Tensor grad_hx = at::empty({batch_size, hidden_size}, workspace.options());
        // has_bias=false: bias outputs stay undefined; the EXEC_NPU_CMD layer
        // converts them to nullptr and the kernel skips the bias slots.
        at::Tensor grad_input_bias = has_bias ? at::empty({3 * hidden_size}, workspace.options()) : at::Tensor();
        at::Tensor grad_hidden_bias = has_bias ? at::empty({3 * hidden_size}, workspace.options()) : at::Tensor();

        EXEC_NPU_CMD(aclnnThnnFusedGruCellBackward, grad_hy, workspace, has_bias, grad_input_gates,
            grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias);
        return std::make_tuple(std::move(grad_input_gates), std::move(grad_hidden_gates), std::move(grad_hx),
            std::move(grad_input_bias), std::move(grad_hidden_bias));
    }
    return thnn_fused_gru_cell_backward_composite(grad_hy, workspace, has_bias);
}
} // namespace op_api
