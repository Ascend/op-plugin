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
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;

at::Tensor construct_quant_sparse_infer_output_tensor(
    const at::Tensor& query, std::string layout_query_str,
    std::string layout_kv_str, const uint64_t &rope_head_dim)
{
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    at::SmallVector<int64_t, SIZE> output_size;
    if (layout_query_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3) - rope_head_dim};
    } else {
        output_size = {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) - rope_head_dim};
    }
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size, query.options().dtype(query.dtype()));

    return output;
}

at::Tensor npu_kv_quant_sparse_flash_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_indices, double scale_value,
    int64_t key_quant_mode, int64_t value_quant_mode,
    const c10::optional<at::Tensor> &key_dequant_scale,
    const c10::optional<at::Tensor> &value_dequant_scale,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_kv,
    int64_t sparse_block_size, c10::string_view layout_query, c10::string_view layout_kv,
    int64_t sparse_mode, int64_t pre_tokens, int64_t next_tokens, int64_t attention_mode, int64_t quant_scale_repo_mode,
    int64_t tile_size, int64_t rope_head_dim)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")

    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    // construct the output tensor
    at::Tensor output = op_api::construct_quant_sparse_infer_output_tensor(
        query, layout_query_str, layout_kv_str, rope_head_dim);
    // convert str
    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnKvQuantSparseFlashAttention, query,
        key, value, sparse_indices, key_dequant_scale, value_dequant_scale, block_table, actual_seq_lengths_query,
        actual_seq_lengths_kv, scale_value, key_quant_mode, value_quant_mode, sparse_block_size, layout_query_ptr,
        layout_kv_ptr, sparse_mode, pre_tokens, next_tokens, attention_mode, quant_scale_repo_mode, tile_size,
        rope_head_dim, output);
    return output;
}

} // namespace op_api