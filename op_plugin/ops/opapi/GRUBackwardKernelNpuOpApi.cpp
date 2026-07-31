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
using npu_preparation = at_npu::native::OpPreparation;

namespace {
// Keep this helper consistent with the one in LSTMBackwardKernelNpuOpApi.cpp.
// Placed in an anonymous namespace to avoid duplicate symbols at link time.
std::vector<at::Tensor> squeeze_chunk_result(const at::TensorList& chunk_result)
{
    std::vector<at::Tensor> squeezed_result;
    for (const auto& chunk : chunk_result) {
        if (chunk.defined()) {
            at::Tensor squeezed_chunk = (chunk.dim() > 0 && chunk.size(0) == 1) ? chunk.squeeze(0) : chunk;
            squeezed_result.push_back(squeezed_chunk);
        } else {
            squeezed_result.push_back(at::Tensor());
        }
    }
    return squeezed_result;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> _gru_npu_backward(
    const at::Tensor &grad_y,
    const at::Tensor &grad_h,
    const at::Tensor &input,
    const at::Tensor &hx,
    const at::TensorList params,
    const at::Tensor &r,
    const at::Tensor &z,
    const at::Tensor &n,
    const at::Tensor &h_n,
    const at::Tensor &h,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    c10::optional<bool> batch_first,
    const c10::optional<at::Tensor> &batch_sizes)
{
    const bool batch_first_1 = batch_first.value_or(false);
    auto out0_shape = op_infer::gru_backward_npu_output_size(input, batch_first_1, batch_sizes);
    auto out1_shape =
        op_infer::gru_backward_npu_h_prev_output_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);

    int64_t D = bidirectional ? 2 : 1;
    int64_t output_format = ACL_FORMAT_ND;
    at::Tensor out0 = npu_preparation::apply_tensor(input, out0_shape);
    at::Tensor out_h_prev = npu_preparation::apply_tensor(input, out1_shape);
    std::vector<at::Tensor> param_list;

    for (int64_t idx = 0; idx < params.size(); ++idx) {
        auto i_tensor = npu_preparation::apply_tensor_with_format(input, params[idx].sizes(), output_format);
        param_list.emplace_back(std::move(i_tensor));
    }

    int64_t list_length = D * num_layers;
    const int64_t split_dim = 0;
    // aclnnGRUBackward accepts hx as a single 3D Tensor of shape
    // [D * num_layers, batch_size, hidden_size] (same as aclnnGRU forward) and
    // slices it internally per layer/direction, so no chunking is needed.
    // The gate/state tensors (r, z, n, h_n, h) are stacked along dim 0 by the
    // forward op, so they must be chunked back into TensorList of length
    // D * num_layers, with each element shape [time_step, batch_size, hidden_size].
    auto r_chunk_origin = at::chunk(r, list_length, split_dim);
    auto z_chunk_origin = at::chunk(z, list_length, split_dim);
    auto n_chunk_origin = at::chunk(n, list_length, split_dim);
    auto h_n_chunk_origin = at::chunk(h_n, list_length, split_dim);
    auto h_chunk_origin = at::chunk(h, list_length, split_dim);

    std::vector<at::Tensor> r_chunk = squeeze_chunk_result(r_chunk_origin);
    std::vector<at::Tensor> z_chunk = squeeze_chunk_result(z_chunk_origin);
    std::vector<at::Tensor> n_chunk = squeeze_chunk_result(n_chunk_origin);
    std::vector<at::Tensor> h_n_chunk = squeeze_chunk_result(h_n_chunk_origin);
    std::vector<at::Tensor> h_chunk = squeeze_chunk_result(h_chunk_origin);

    at::TensorList r_list = at::TensorList(r_chunk);
    at::TensorList z_list = at::TensorList(z_chunk);
    at::TensorList n_list = at::TensorList(n_chunk);
    at::TensorList h_n_list = at::TensorList(h_n_chunk);
    at::TensorList h_list_ = at::TensorList(h_chunk);

    at::TensorList param_list_ = at::TensorList(param_list);
    c10::optional<at::Tensor> batch_sizes_opt = batch_sizes;
    // The autograd engine may materialise a None optional<Tensor> as an empty
    // (shape [0]) defined tensor.  Normalise it back to nullopt so the CANN
    // operator sees the regular (non-packed) path.
    if (batch_sizes_opt.has_value() && batch_sizes_opt.value().numel() == 0) {
        batch_sizes_opt = c10::nullopt;
    }

    // aclnnGRUBackward requires dy/dh to be non-null tensors (CheckNotNull rejects
    // undefined tensors with ACLNN_ERR_PARAM_NULLPTR). When autograd only flows
    // through one of output_y/output_h, the other grad is undefined; fall back to
    // a zero tensor with the correct shape for the forward counterpart so the
    // zero contribution does not alter the gradient computation.
    // dh shape per aclnn spec: [numLayers * D, batch_size, hidden_size], same as hx.
    // dy shape per aclnn spec: same as forward output_y whose last dim is D * hidden_size,
    // NOT input_size.  Using zeros_like(input) would produce the wrong shape when
    // input_size != D * hidden_size.
    at::Tensor grad_y_real = grad_y;
    if (!grad_y_real.defined()) {
        auto dy_shape = op_infer::gru_npu_output_size(input, params, bidirectional, batch_first_1, batch_sizes_opt);
        grad_y_real = at::zeros(dy_shape, input.options());
    }
    at::Tensor grad_h_real = grad_h.defined() ? grad_h : at::zeros_like(hx);

    EXEC_NPU_CMD(
        aclnnGRUBackward,
        input,
        params,
        hx,
        grad_y_real,
        grad_h_real,
        r_list,
        z_list,
        n_list,
        h_n_list,
        h_list_,
        batch_sizes_opt,
        has_biases,
        num_layers,
        bidirectional,
        batch_first_1,
        out0,
        out_h_prev,
        param_list_);

    return std::make_tuple(
        out0,
        out_h_prev,
        param_list_.vec());
}

} // namespace op_api
