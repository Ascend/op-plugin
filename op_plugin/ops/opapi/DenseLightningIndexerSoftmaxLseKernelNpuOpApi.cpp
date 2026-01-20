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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;

std::tuple<at::Tensor, at::Tensor> npu_dense_lightning_indexer_softmax_lse_symint(
    const at::Tensor &query_index,
    const at::Tensor &key_index,
    const at::Tensor &weights,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_klen,
    c10::optional<c10::string_view> layout,
    c10::optional<int64_t> sparse_mode,
    c10::optional<int64_t> pre_tokens,
    c10::optional<int64_t> next_tokens)
{
    TORCH_CHECK(query_index.dim() == DIMENSION_3D || query_index.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query_index.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key_index.dim() == DIMENSION_3D || key_index.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key_index.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));

    c10::string_view layout_str = layout.value_or("BSND");
    char *layout_ptr = const_cast<char *>(layout_str.data());
    int64_t sparse_mode_const = sparse_mode.value_or(3);
    int64_t pre_tokens_const = pre_tokens.value_or(9223372036854775807);
    int64_t next_tokens_const = next_tokens.value_or(9223372036854775807);

    c10::SmallVector<int64_t, op_infer::SIZE> output_size;
    std::string input_layout = std::string(layout_str);
    if (input_layout == "TND") {
        output_size = {key_index.size(1), query_index.size(0)};
    } else {
        output_size = {query_index.size(0), key_index.size(2), query_index.size(1)};
    }

    at::Tensor softmax_max_out = OpPreparation::apply_tensor_without_format(output_size, query_index.options().dtype(at::kFloat));
    at::Tensor softmax_sum_out = OpPreparation::apply_tensor_without_format(output_size, query_index.options().dtype(at::kFloat));

    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnDenseLightningIndexerSoftmaxLse, query_index, key_index, weights,
        actual_seq_qlen, actual_seq_klen, layout_ptr, sparse_mode_const, pre_tokens_const,
        next_tokens_const, softmax_max_out, softmax_sum_out);

    return std::make_tuple(softmax_max_out, softmax_sum_out);
}
}
