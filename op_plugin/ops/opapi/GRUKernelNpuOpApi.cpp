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
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool IsBf16Tensor(const at::Tensor& t)
{
    return t.defined() && t.scalar_type() == at::kBFloat16;
}

inline bool HasBf16Tensor(const at::Tensor& input, const at::Tensor& hx, const at::TensorList params)
{
    if (IsBf16Tensor(input)) {
        return true;
    }
    if (IsBf16Tensor(hx)) {
        return true;
    }
    for (const auto& t : params) {
        if (IsBf16Tensor(t)) {
            return true;
        }
    }
    return false;
}

inline bool HasMixedFloatDtype(const at::Tensor& input, const at::Tensor& hx, const at::TensorList params)
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

    if (mismatch_with_input(hx)) {
        return true;
    }
    for (const auto& t : params) {
        if (mismatch_with_input(t)) {
            return true;
        }
    }
    return false;
}

inline bool ShouldFallbackToAclOp(const at::Tensor& input, const at::Tensor& hx, const at::TensorList params)
{
    return HasBf16Tensor(input, hx, params) || HasMixedFloatDtype(input, hx, params);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> _gru_npu(
    const at::Tensor &input,
    const at::Tensor &hx,
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
    auto out0_shape = op_infer::gru_npu_output_size(input, params, bidirectional, batch_first_1, batch_sizes);
    auto out1_shape = op_infer::gru_npu_hy_output_size(input, params, num_layers, bidirectional, batch_first_1, batch_sizes);
    auto gate_shape = op_infer::gru_npu_gate_output_size(input, params, train, bidirectional, batch_first_1, batch_sizes);

    int64_t D = bidirectional ? 2 : 1;
    int64_t output_format = ACL_FORMAT_ND;
    int64_t list_length = D * num_layers;

    at::Tensor out0 = npu_preparation::apply_tensor(input, out0_shape);
    at::Tensor out1 = npu_preparation::apply_tensor(input, out1_shape);

    std::vector<at::Tensor> r_list;
    std::vector<at::Tensor> z_list;
    std::vector<at::Tensor> n_list;
    std::vector<at::Tensor> hn_list;
    std::vector<at::Tensor> h_list;

    at::TensorList r_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList z_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList n_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList hn_list_ = c10::ArrayRef<at::Tensor>();
    at::TensorList h_list_ = c10::ArrayRef<at::Tensor>();

    if (train) {
        r_list.reserve(list_length);
        z_list.reserve(list_length);
        n_list.reserve(list_length);
        hn_list.reserve(list_length);
        h_list.reserve(list_length);

        for (int64_t idx = 0; idx < list_length; ++idx) {
            r_list.emplace_back(npu_preparation::apply_tensor_with_format(input, gate_shape, output_format));
            z_list.emplace_back(npu_preparation::apply_tensor_with_format(input, gate_shape, output_format));
            n_list.emplace_back(npu_preparation::apply_tensor_with_format(input, gate_shape, output_format));
            hn_list.emplace_back(npu_preparation::apply_tensor_with_format(input, gate_shape, output_format));
            h_list.emplace_back(npu_preparation::apply_tensor_with_format(input, gate_shape, output_format));
        }
        r_list_ = at::TensorList(r_list);
        z_list_ = at::TensorList(z_list);
        n_list_ = at::TensorList(n_list);
        hn_list_ = at::TensorList(hn_list);
        h_list_ = at::TensorList(h_list);
    }

    c10::optional<at::Tensor> batch_sizes_opt = batch_sizes;
    // Normalise an empty (shape [0]) batch_sizes tensor back to nullopt so the
    // CANN operator takes the regular (non-packed) path.  The autograd engine
    // may materialise a None optional<Tensor> as an empty defined tensor.
    if (batch_sizes_opt.has_value() && batch_sizes_opt.value().numel() == 0) {
        batch_sizes_opt = c10::nullopt;
    }
    EXEC_NPU_CMD(
        aclnnGRU,
        input,
        params,
        hx,
        batch_sizes_opt,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first_1,
        out0,
        out1,
        r_list_,
        z_list_,
        n_list_,
        hn_list_,
        h_list_);

    // at::stack allocates the output with the NPU default storage format (NCHW),
    // but aclnnGRUBackward requires every input to be ACL_FORMAT_ND. Cast the
    // stacked gate/state tensors back to ND so the tensors saved for backward
    // satisfy the reverse operator's format constraint.
    auto stack_as_nd = [](const std::vector<at::Tensor>& vec) -> at::Tensor {
        if (vec.empty()) {
            return at::Tensor();
        }
        at::Tensor stacked = at::stack(vec, 0);
        return at_npu::native::npu_format_cast(stacked, ACL_FORMAT_ND);
    };
    at::Tensor r_tensor = stack_as_nd(r_list_.vec());
    at::Tensor z_tensor = stack_as_nd(z_list_.vec());
    at::Tensor n_tensor = stack_as_nd(n_list_.vec());
    at::Tensor hn_tensor = stack_as_nd(hn_list_.vec());
    at::Tensor h_tensor = stack_as_nd(h_list_.vec());

    return std::make_tuple(out0, out1, r_tensor, z_tensor, n_tensor, hn_tensor, h_tensor);
}

std::tuple<at::Tensor, at::Tensor> gru(
    const at::Tensor &input_val, const at::Tensor &hx, at::TensorList params,
    bool has_biases, int64_t num_layers, double dropout, bool train,
    bool bidirectional, bool batch_first)
{
    // If bf16 or mixed dtype, fallback to acl_op
    if (ShouldFallbackToAclOp(input_val, hx, params)) {
        return acl_op::gru(input_val, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    }

    DO_COMPATIBILITY(aclnnGRU, acl_op::gru(input_val, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first));
    auto output = at_npu::native::custom_ops::_gru_npu(input_val, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
    return std::make_tuple(std::get<0>(output), std::get<1>(output)); // 0 for output_y, 1 for output_h
}

std::tuple<at::Tensor, at::Tensor> gru(
    const at::Tensor& data, const at::Tensor& batch_sizes, const at::Tensor& hx,
    at::TensorList params, bool has_biases,
    int64_t num_layers, double dropout,
    bool train, bool bidirectional)
{
    // Packed-sequence (gru.data) path is only supported via aclnnGRU; the
    // acl_op DynamicGRUV2 implementation does not support batch_sizes, so there
    // is no fallback for bf16/mixed dtypes here.
    auto output = at_npu::native::custom_ops::_gru_npu(data, hx, params, has_biases, num_layers, dropout, train, bidirectional, false, batch_sizes);
    return std::make_tuple(std::get<0>(output), std::get<1>(output)); // 0 for output_y, 1 for output_h
}
}
