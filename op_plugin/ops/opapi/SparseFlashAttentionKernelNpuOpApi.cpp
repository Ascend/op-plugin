// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;

using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor construct_sparse_flash_attention_output_tensor(
    const at::Tensor& query, std::string layout)
{
    TORCH_CHECK(layout == "BSND" || layout == "TND", "The layout of query only support BSND and TND, but got ",
                layout, OPS_ERROR(ErrCode::PARAM));
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i), OPS_ERROR(ErrCode::PARAM));
    }
    if (layout == "TND") {
        TORCH_CHECK(query.dim() == DIM_3,
                    "When the layout of query is TND, the query dimension must be 3, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    } else {
        TORCH_CHECK(query.dim() == DIM_4,
                    "When the layout of query is BSND, the query dimension must be 4, but got ",
                    query.dim(), OPS_ERROR(ErrCode::PARAM));
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));

    return output;
}
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, int64_t sparse_block_size,
    c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens,
    int64_t attention_mode, bool return_softmax_lse)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.", OPS_ERROR(ErrCode::PARAM));

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor sparse_flash_attention_output = construct_sparse_flash_attention_output_tensor(
        query, layout_query_str);
    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::SmallVector<int64_t, SIZE> softmax_max_size;
    at::SmallVector<int64_t, SIZE> softmax_sum_size;
    if (query.dim() == DIM_3) {
        softmax_max_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
        softmax_sum_size = {key.size(1), query.size(0), query.size(1) / key.size(1)};
    } else {
        softmax_max_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
        softmax_sum_size = {query.size(0), key.size(2), query.size(1), query.size(2) / key.size(2)};
    }
    softmax_max = at::empty(softmax_max_size, query.options().dtype(at::kFloat));
    softmax_sum = at::empty(softmax_sum_size, query.options().dtype(at::kFloat));
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnSparseFlashAttention, query,
        key, value, sparse_indices, scale_value, block_table, actual_seq_lengths_query,
        actual_seq_lengths_kv, query_rope, key_rope, sparse_block_size,
        layout_query_ptr, layout_kv_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse,
        sparse_flash_attention_output, softmax_max, softmax_sum);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(sparse_flash_attention_output, softmax_max, softmax_sum);
}

} // namespace op_api