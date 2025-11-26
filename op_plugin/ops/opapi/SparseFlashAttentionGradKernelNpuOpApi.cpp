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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &sparse_indices,
    const at::Tensor &d_out,
    const at::Tensor &out,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    double scale_value,
    int64_t sparse_block_size,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    const c10::optional<at::Tensor> &actual_seq_kvlen,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens,
    c10::optional<int64_t> attention_mode)
{
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    const at::Tensor &ac_seq_qlen = actual_seq_qlen.value_or(at::Tensor());
    const at::Tensor &ac_seq_kvlen = actual_seq_kvlen.value_or(at::Tensor());
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
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    at::Tensor d_query = OpPreparation::apply_tensor_without_format(query);
    at::Tensor d_key = OpPreparation::apply_tensor_without_format(key);
    at::Tensor d_value = OpPreparation::apply_tensor_without_format(value);
    at::Tensor d_query_rope;
    at::Tensor d_key_rope;
    if (query_rope_const.defined()) {
        d_query_rope = OpPreparation::apply_tensor_without_format(query_rope_const);
    } else {
        d_query_rope = at::empty({0}, query.options());
    }
    if (key_rope_const.defined()) {
        d_key_rope = OpPreparation::apply_tensor_without_format(key_rope_const);
    } else {
        d_key_rope = at::empty({0}, key.options());
    }

    c10::string_view layout_str_view = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str_view.data());

    const int64_t sparse_mode_const = sparse_mode.value_or(3);
    const int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    const int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    const bool deterministic_const = true;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnSparseFlashAttentionGrad, query, key, value, sparse_indices, d_out,
        out, softmax_max, softmax_sum, ac_seq_qlen, ac_seq_kvlen, query_rope_const, key_rope_const,
        scale_value, sparse_block_size, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const,
        deterministic_const, d_query, d_key, d_value, d_query_rope, d_key_rope);

    return std::make_tuple(d_query, d_key, d_value, d_query_rope, d_key_rope);
}
}
