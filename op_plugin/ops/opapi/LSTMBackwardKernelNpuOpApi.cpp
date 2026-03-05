// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>> _lstm_backward_npu(
    const at::Tensor &grad_y,
    const at::Tensor &grad_hy,
    const at::Tensor &grad_cy,
    const at::Tensor &input,
    const at::TensorList hx,
    const at::TensorList params,
    const at::Tensor &i,
    const at::Tensor &j,
    const at::Tensor &f,
    const at::Tensor &o,
    const at::Tensor &h,
    const at::Tensor &c,
    const at::Tensor &tanhc,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    c10::optional<bool> batch_first,
    const c10::optional<at::Tensor> &batch_sizes)
{
    const bool batch_first_1 = batch_first.value_or(false);
    auto out0_shape = op_infer::lstm_backward_npu_output_size(input, batch_first_1, batch_sizes);
    auto out1_shape = op_infer::lstm_backward_npu_hc_prev_output_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);

    int64_t per_layer_params = has_biases ? 4 : 2;
    int64_t D = bidirectional ? 2 : 1;
    int64_t output_format = ACL_FORMAT_ND;
    at::Tensor out0 = npu_preparation::apply_tensor(input, out0_shape);
    at::Tensor out_h_prev = npu_preparation::apply_tensor(input, out1_shape);
    at::Tensor out_c_prev = npu_preparation::apply_tensor(input, out1_shape);
    std::vector<at::Tensor> param_list;

    for (int64_t idx = 0; idx < params.size(); ++idx) {
        auto i_tensor = npu_preparation::apply_tensor_with_format(input, params[idx].sizes(), output_format);
        param_list.emplace_back(std::move(i_tensor));
    }

    int64_t list_length = D * num_layers;

    const int64_t split_dim = 0;
    auto i_chunk_origin = at::chunk(i, list_length, split_dim);
    auto j_chunk_origin = at::chunk(j, list_length, split_dim);
    auto f_chunk_origin = at::chunk(f, list_length, split_dim);
    auto o_chunk_origin = at::chunk(o, list_length, split_dim);
    auto h_chunk_origin = at::chunk(h, list_length, split_dim);
    auto c_chunk_origin = at::chunk(c, list_length, split_dim);
    auto tanhc_chunk_origin = at::chunk(tanhc, list_length, split_dim);

    std::vector<at::Tensor> i_chunk = squeeze_chunk_result(i_chunk_origin);
    std::vector<at::Tensor> j_chunk = squeeze_chunk_result(j_chunk_origin);
    std::vector<at::Tensor> f_chunk = squeeze_chunk_result(f_chunk_origin);
    std::vector<at::Tensor> o_chunk = squeeze_chunk_result(o_chunk_origin);
    std::vector<at::Tensor> h_chunk = squeeze_chunk_result(h_chunk_origin);
    std::vector<at::Tensor> c_chunk = squeeze_chunk_result(c_chunk_origin);
    std::vector<at::Tensor> tanhc_chunk = squeeze_chunk_result(tanhc_chunk_origin);

    at::TensorList i_list = at::TensorList(i_chunk);
    at::TensorList j_list = at::TensorList(j_chunk);
    at::TensorList f_list = at::TensorList(f_chunk);
    at::TensorList o_list = at::TensorList(o_chunk);
    at::TensorList h_list = at::TensorList(h_chunk);
    at::TensorList c_list = at::TensorList(c_chunk);
    at::TensorList tanhc_list = at::TensorList(tanhc_chunk);

    at::TensorList param_list_ = at::TensorList(param_list);
    std::nullptr_t temp_nullptr = nullptr;
    EXEC_NPU_CMD(
        aclnnLstmBackward,
        input,
        hx,
        params,
        grad_y,
        grad_hy,
        grad_cy,
        i_list,
        j_list,
        f_list,
        o_list,
        h_list,
        c_list,
        tanhc_list,
        batch_sizes,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first_1,
        temp_nullptr,
        out0,
        out_h_prev,
        out_c_prev,
        param_list_);

    return std::make_tuple(
        out0,
        out_h_prev,
        out_c_prev,
        param_list_.vec());
}
}
