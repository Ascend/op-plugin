// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> _thnn_fused_gru_cell(
    const at::Tensor& input_gates, const at::Tensor& hidden_gates, const at::Tensor& hx, const c10::optional<at::Tensor>& input_bias, const c10::optional<at::Tensor>& hidden_bias)
{
    at::Tensor igates_b = (input_bias.has_value() && input_bias.value().defined()) ? input_gates + input_bias.value() : input_gates;
    at::Tensor hgates_b = (hidden_bias.has_value() && hidden_bias.value().defined()) ? hidden_gates + hidden_bias.value() : hidden_gates;

    auto chunked_igates = igates_b.unsafe_chunk(3, 1); // [reset, input, new]
    auto chunked_hgates = hgates_b.unsafe_chunk(3, 1);

    const auto reset_gate = chunked_hgates[0].add_(chunked_igates[0]).sigmoid_();
    const auto input_gate = chunked_hgates[1].add_(chunked_igates[1]).sigmoid_();
    const auto new_gate = chunked_igates[2].add(chunked_hgates[2].mul_(reset_gate)).tanh_();
    auto hy = (hx - new_gate).mul_(input_gate).add_(new_gate);

    // 返回新状态和空 workspace（保持接口兼容）
    return std::make_tuple(hy, at::Tensor());
}
}
