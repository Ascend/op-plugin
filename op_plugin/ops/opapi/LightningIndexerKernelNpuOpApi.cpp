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
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> construct_lightning_indexer_output_tensor(const at::Tensor& query,
    const at::Tensor& key, const c10::optional<at::Tensor> &actual_seq_lengths_query, int64_t sparse_count,
    std::string query_layout_str, std::string key_layout_str, bool return_value)
{
    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), key.size(DIM_2), sparse_count};
    } else {
        int n_dim_index = 0;
        n_dim_index = (key_layout_str == "TND") ? DIM_1 : DIM_2;
        output_size = {query.size(DIM_0), key.size(n_dim_index), sparse_count};
    }
    at::Tensor sparse_indices_out = npu_preparation::apply_tensor_without_format(output_size, at::kInt);
    at::Tensor sparse_values_out;
    if (return_value) {
        sparse_values_out = npu_preparation::apply_tensor_without_format(output_size, query.dtype());
    } else {
        sparse_values_out = npu_preparation::apply_tensor_without_format({0}, query.dtype());
    }

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

std::tuple<at::Tensor, at::Tensor> npu_lightning_indexer(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, c10::string_view layout_query,
    c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode,
    int64_t pre_tokens, int64_t next_tokens, bool return_value)
{
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.")
    TORCH_CHECK(key.numel() > 0, "Tensor key is empty.")

    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> lightning_indexer_output = op_api::construct_lightning_indexer_output_tensor(
        query, key, actual_seq_lengths_query, sparse_count, query_layout_str, key_layout_str, return_value);
    at::Tensor sparse_indices_out = std::get<0>(lightning_indexer_output);
    at::Tensor sparse_values_out = std::get<1>(lightning_indexer_output);
    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnLightningIndexer, query,
        key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table,
        query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode, pre_tokens, next_tokens,
        return_value, sparse_indices_out, sparse_values_out);

    return std::tuple<at::Tensor, at::Tensor>(sparse_indices_out, sparse_values_out);
}

} // namespace op_api