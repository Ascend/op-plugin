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

    at::SmallVector<int64_t, op_infer::SIZE> grad_bias_size = {workspace.size(1)};
    at::Tensor grad_gates = npu_preparation::apply_tensor(workspace);
    at::Tensor grad_cx = npu_preparation::apply_tensor(cx);
    at::Tensor grad_bias = npu_preparation::apply_tensor_without_format(grad_bias_size, c10::dtype(cx.scalar_type()));

    EXEC_NPU_CMD(aclnnThnnFusedLstmCellBackward, grad_hy_opt, grad_cy_opt, cx, cy, workspace, has_bias, grad_gates,
        grad_cx, grad_bias);
    return std::tie(grad_gates, grad_cx, grad_bias);
}
} // namespace op_api