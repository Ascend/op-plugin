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

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_lightning_indexer_grad(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &dy,
    const at::Tensor &sparse_indices,
    const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens)
{
    const at::Tensor &actual_seq_lengths_query_const = actual_seq_lengths_query.value_or(at::Tensor());
    const at::Tensor &actual_seq_lengths_key_const = actual_seq_lengths_key.value_or(at::Tensor());

    at::Tensor d_query = OpPreparation::apply_tensor_without_format(query);
    at::Tensor d_key = OpPreparation::apply_tensor_without_format(key);
    at::Tensor d_weights = OpPreparation::apply_tensor_without_format(weights);

    c10::string_view layout_str_view = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str_view.data());
    const int64_t sparse_mode_const = sparse_mode.value_or(0);
    const int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    const int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);
    const int64_t head_num = 0;
    const bool deterministic = true;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnLightningIndexerGrad, query, key, dy, sparse_indices, weights,
        actual_seq_lengths_query_const, actual_seq_lengths_key_const,
        head_num, layout_ptr, sparse_mode_const, pre_tokens_const, next_tokens_const, deterministic,
        d_query, d_key, d_weights);

    return std::make_tuple(d_query, d_key, d_weights);
}
}