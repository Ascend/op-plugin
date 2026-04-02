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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _lstm_npu(
    const at::Tensor &input,
    const at::TensorList hx,
    const at::TensorList params,
    bool has_biases,
    int64_t num_layers,
    double dropout,
    bool train,
    bool bidirectional,
    c10::optional<bool> batch_first,
    const c10::optional<at::Tensor> &batch_sizes)
{
    const bool batch_first_1 = batch_first.value_or(false);
    auto out0_shape = op_infer::lstm_npu_output_size(input, params, bidirectional, batch_first_1, batch_sizes);
    auto out1_shape = op_infer::lstm_npu_output1_2_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);
    auto out2_shape = op_infer::lstm_npu_output1_2_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);
    auto ijfo_hc_tanhc_shapes = op_infer::lstm_npu_ijfo_hc_tanhc_output_size(input, params, num_layers, train, bidirectional, batch_first_1, batch_sizes);

    int64_t D = bidirectional ? 2 : 1;
    int64_t output_format = ACL_FORMAT_ND;

    at::Tensor out0 = npu_preparation::apply_tensor(input, out0_shape);
    at::Tensor out1 = npu_preparation::apply_tensor(input, out1_shape);
    at::Tensor out2 = npu_preparation::apply_tensor(input, out2_shape);
    int64_t list_length = D * num_layers;
    std::vector<at::Tensor> i_list;
    std::vector<at::Tensor> j_list;
    std::vector<at::Tensor> f_list;
    std::vector<at::Tensor> o_list;
    std::vector<at::Tensor> tanh_list;
    std::vector<at::Tensor> h_list;
    std::vector<at::Tensor> c_list;
    h_list.reserve(list_length);
    c_list.reserve(list_length);

    at::TensorList i_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList j_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList f_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList o_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList tanh_list_ = c10::ArrayRef<at::Tensor>();
    if (train) {
        i_list.reserve(list_length);
        j_list.reserve(list_length);
        f_list.reserve(list_length);
        o_list.reserve(list_length);
        tanh_list.reserve(list_length);

        for (int64_t idx = 0; idx < list_length; ++idx) {
            auto i_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            i_list.emplace_back(std::move(i_tensor));

            auto j_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            j_list.emplace_back(std::move(j_tensor));

            auto f_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            f_list.emplace_back(std::move(f_tensor));

            auto o_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            o_list.emplace_back(std::move(o_tensor));

            auto tanh_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            tanh_list.emplace_back(std::move(tanh_tensor));

            auto h_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            h_list.emplace_back(std::move(h_tensor));

            auto c_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            c_list.emplace_back(std::move(c_tensor));
        }
        i_list_ = at::TensorList(i_list);
        j_list_ = at::TensorList(j_list);
        f_list_ = at::TensorList(f_list);
        o_list_ = at::TensorList(o_list);
        tanh_list_ = at::TensorList(tanh_list);
    } else {
        for (int64_t idx = 0; idx < list_length; ++idx) {
            auto h_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            h_list.emplace_back(std::move(h_tensor));
            auto c_tensor = npu_preparation::apply_tensor_with_format(input, ijfo_hc_tanhc_shapes, output_format);
            c_list.emplace_back(std::move(c_tensor));
        }
    }

    at::TensorList h_list_ = at::TensorList(h_list);
    at::TensorList c_list_ = at::TensorList(c_list);

    EXEC_NPU_CMD(
        aclnnLSTM,
        input,
        params,
        hx,
        batch_sizes,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first_1,
        out0,
        out1,
        out2,
        i_list_,
        j_list_,
        f_list_,
        o_list_,
        h_list_,
        c_list_,
        tanh_list_);

    at::Tensor i_tensor = i_list_.vec().empty() ? at::Tensor() : at::stack(i_list_.vec(), 0);
    at::Tensor j_tensor = j_list_.vec().empty() ? at::Tensor() : at::stack(j_list_.vec(), 0);
    at::Tensor f_tensor = f_list_.vec().empty() ? at::Tensor() : at::stack(f_list_.vec(), 0);
    at::Tensor o_tensor = o_list_.vec().empty() ? at::Tensor() : at::stack(o_list_.vec(), 0);
    at::Tensor tanh_tensor = tanh_list_.vec().empty() ? at::Tensor() : at::stack(tanh_list_.vec(), 0);
    at::Tensor h_tensor = h_list_.vec().empty() ? at::Tensor() : at::stack(h_list_.vec(), 0);
    at::Tensor c_tensor = c_list_.vec().empty() ? at::Tensor() : at::stack(c_list_.vec(), 0);

    return std::make_tuple(
        out0,
        out1,
        out2,
        i_tensor,
        j_tensor,
        f_tensor,
        o_tensor,
        h_tensor,
        c_tensor,
        tanh_tensor);
}

inline bool IsBf16Tensor(const at::Tensor& t)
{
    return t.defined() && t.scalar_type() == at::kBFloat16;
}

inline bool HasBf16Tensor(const at::Tensor& input, const at::TensorList hx, const at::TensorList params)
{
    if (IsBf16Tensor(input)) {
        return true;
    }
    for (const auto& t : hx) {
        if (IsBf16Tensor(t)) {
            return true;
        }
    }
    for (const auto& t : params) {
        if (IsBf16Tensor(t)) {
            return true;
        }
    }
    return false;
}

inline bool HasMixedFloatDtype(const at::Tensor& input, const at::TensorList hx, const at::TensorList params)
{
    if (!input.defined() || !at::isFloatingType(input.scalar_type())) {
        return false;
    }

    const auto ref_dtype = input.scalar_type();

    auto mismatch_with_input = [ref_dtype](const at::Tensor& t) {
        return t.defined() &&
               at::isFloatingType(t.scalar_type()) &&
               t.scalar_type() != ref_dtype;
    };

    for (const auto& t : hx) {
        if (mismatch_with_input(t)) {
            return true;
        }
    }
    for (const auto& t : params) {
        if (mismatch_with_input(t)) {
            return true;
        }
    }
    return false;
}

inline bool ShouldFallbackToAclOp(const at::Tensor& input, const at::TensorList hx, const at::TensorList params)
{
    return HasBf16Tensor(input, hx, params) || HasMixedFloatDtype(input, hx, params);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& input, const at::TensorList hx,
    const at::TensorList params, bool has_biases,
    int64_t num_layers, double dropout,
    bool train, bool bidirectional, bool batch_first)
{
    // If bf16 or mixed dtype, fallback to acl_op
    if (ShouldFallbackToAclOp(input, hx, params)) {
        return acl_op::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    }

    DO_COMPATIBILITY(aclnnLSTM, acl_op::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first));
    auto output = at_npu::native::custom_ops::_lstm_npu(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_tuple(std::get<0>(output), std::get<1>(output), std::get<2>(output)); // 0 for output_y, 1 for output_h, 2 for output_c
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(
    const at::Tensor& data, const at::Tensor& batch_sizes, const at::TensorList hx,
    const at::TensorList params, bool has_biases,
    int64_t num_layers, double dropout,
    bool train, bool bidirectional)
{
    // If bf16 or mixed dtype, fallback to acl_op
    if (ShouldFallbackToAclOp(data, hx, params)) {
        return acl_op::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
    }

    DO_COMPATIBILITY(aclnnLSTM, acl_op::lstm(data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional));
    auto output = at_npu::native::custom_ops::_lstm_npu(data, hx, params, has_biases, num_layers, dropout, train, bidirectional, false, batch_sizes);
    return std::make_tuple(std::get<0>(output), std::get<1>(output), std::get<2>(output)); // 0 for output_y, 1 for output_h, 2 for output_c
}
}