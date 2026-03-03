// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> _thnn_fused_lstm_cell_backward_impl(
    const std::optional<at::Tensor>& grad_hy_opt, const std::optional<at::Tensor>& grad_cy_opt,
    const at::Tensor& cx, const at::Tensor& cy, const at::Tensor& workspace, bool has_bias)
{
    // check input tensor
    const unsigned int dim2D = 2;
    TORCH_CHECK(cx.dim() == dim2D && cy.dim() == dim2D && workspace.dim() == dim2D, \
        "cx, cy and workspace must be a 2D Tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(cx.sizes() == cy.sizes(), "cx and cy must be the same size", OPS_ERROR(ErrCode::PARAM));

    c10::optional<at::Tensor> grad_cy;
    if (!grad_cy_opt.has_value()) {
        grad_cy = c10::optional<at::Tensor>(at::Tensor());
    } else {
        grad_cy = grad_cy_opt;
    }

    at::SmallVector<int64_t, op_infer::SIZE> grad_bias_size = {workspace.size(1)};
    at::Tensor grad_gates = npu_preparation::apply_tensor(workspace);
    at::Tensor grad_cx = npu_preparation::apply_tensor(cx);
    at::Tensor grad_bias = npu_preparation::apply_tensor_without_format(grad_bias_size, c10::dtype(cx.scalar_type()));

    EXEC_NPU_CMD(aclnnThnnFusedLstmCellBackward, grad_hy_opt, grad_cy, cx, cy, workspace, has_bias, grad_gates,
        grad_cx, grad_bias);
    return std::tie(grad_gates, grad_cx, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _thnn_fused_lstm_cell(const at::Tensor& i_gates,
    const at::Tensor& h_gates, const at::Tensor& c, const c10::optional<at::Tensor> &input_bias_opt,
    const c10::optional<at::Tensor> &hidden_bias_opt)
{
    auto dtype = i_gates.dtype();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kHalf, "lstm_cell input_gates must be float or half")
    TORCH_CHECK(h_gates.dtype() == dtype, "lstm_cell input_gates and hidden_gates must have same dtype");
    TORCH_CHECK(c.dtype() == dtype, "lstm_cell input_gates and c must have same dtype");
    at::Tensor storage = at::empty_like(i_gates, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor hout = at::empty_like(c, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    at::Tensor cout = at::empty_like(c, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    // Call aclnnLSTMCell operator
    EXEC_NPU_CMD(aclnnThnnFusedLstmCell, i_gates, h_gates, c, input_bias_opt, hidden_bias_opt, hout, cout, storage);
    return std::make_tuple(std::move(hout), std::move(cout), std::move(storage));
}

std::tuple<at::Tensor, at::Tensor> lstm_cell(const at::Tensor &input, at::TensorList hx, const at::Tensor &w_ih,
    const at::Tensor &w_hh, const c10::optional<at::Tensor> &b_ih_opt, const c10::optional<at::Tensor> &b_hh_opt)
{
    at::Tensor h = hx[0];
    at::Tensor c = hx[1];
    auto igates = at::matmul(input, w_ih.t());
    auto hgates = at::matmul(h, w_hh.t());
    auto result = at::_thnn_fused_lstm_cell(igates, hgates, c, b_ih_opt, b_hh_opt);
    return std::make_tuple(std::move(std::get<0>(result)), std::move(std::get<1>(result)));
}
} // namespace op_api
