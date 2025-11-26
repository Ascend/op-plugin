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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_lightning_indexer_grad_kl_loss_symint(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &query_index,
    const at::Tensor &key_index,
    const at::Tensor &weights,
    const at::Tensor &sparse_indices,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    double scale_value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_klen,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens)
{
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    auto actual_seq_qlen_const = actual_seq_qlen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_qlen.value()) : at::IntArrayRef{};
    auto actual_seq_klen_const = actual_seq_klen.has_value() ? c10::asIntArrayRefUnchecked(actual_seq_klen.value()) : at::IntArrayRef{};
    c10::string_view layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    bool deterministic_const = true;
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    at::Tensor d_query_index = OpPreparation::apply_tensor_without_format(query_index);
    at::Tensor d_key_index = OpPreparation::apply_tensor_without_format(key_index);
    at::Tensor d_weights = OpPreparation::apply_tensor_without_format(weights);
    at::Tensor loss = OpPreparation::apply_tensor_without_format({1}, query.options().dtype(at::kFloat));

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnSparseLightningIndexerGradKLLoss, query, key, query_index, key_index, weights,
        sparse_indices, softmax_max, softmax_sum, query_rope_const, key_rope_const, actual_seq_qlen_const,
        actual_seq_klen_const, scale_value, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic_const,
        d_query_index, d_key_index, d_weights, loss);

    return std::make_tuple(d_query_index, d_key_index, d_weights, loss);
}
}
