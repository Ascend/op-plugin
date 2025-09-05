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

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/custom_functions/opapi/update_op_api_common.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
const static int64_t DIM_0 = 0;
const static int64_t DIM_1 = 1;
const static int64_t DIM_2 = 2;
const static int64_t DIM_3 = 3;
const static int64_t DIM_4 = 4;
const static int64_t PA_BBH_DIMS = 3;
const static int64_t PA_BNBD_DIMS = 4;
const static int64_t PA_NZ_DIMS = 5;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> construct_fia_output_tensor_v2(
    const at::Tensor &query,
    const at::Tensor &value,
    std::string input_layout_str,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &block_table,
    int64_t num_query_heads,
    int64_t num_key_value_heads, // 增加kvhead用于计算BBH情况下的D
    bool return_softmax_lse,
    const c10::optional<at::Tensor> &query_rope)
{
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    at::Tensor tmp_output = npu_preparation::apply_tensor_without_format(query);
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = OpPreparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BNSD_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "BSND_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_2), query.size(DIM_0), query.size(DIM_1), query.size(DIM_3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH_NBSD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {num_query_heads, query.size(DIM_0), query.size(DIM_1), query.size(DIM_2) / num_query_heads},
            query.options().dtype(query.dtype()));
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "TND_NTD") {
        tmp_output = OpPreparation::apply_tensor_without_format(
            {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)},
            query.options().dtype(query.dtype()));
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.size(DIM_0);
        qsSize = query.size(DIM_2);
    } else if (input_layout_str == "TND") {
        int64_t kv_dim = value.dim();
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2) * value.size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = OpPreparation::apply_tensor_without_format(
                {query.size(DIM_0), query.size(DIM_1), value.size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    } else if (input_layout_str == "NTD_TND") {
        int64_t kv_dim = value.dim();
        if (kv_dim == 0) {
            kv_dim = query.dim();
        }
        if (block_table.has_value()) { // pa场景
            if (kv_dim == PA_BBH_DIMS) { // BBH的情况下，D = H / N
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) / num_key_value_heads},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_BNBD_DIMS) { // BNBD情况下取D
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_3)},
                    query.options().dtype(query.dtype()));
            } else if (kv_dim == PA_NZ_DIMS) { // blockNum, N, D / 16, blockSize, 16取DIM2*DIM4
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2) * value.size(DIM_4)},
                    query.options().dtype(query.dtype()));
            } else {
                tmp_output = OpPreparation::apply_tensor_without_format(
                    {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                    query.options().dtype(query.dtype()));
            }
        } else {
            tmp_output = OpPreparation::apply_tensor_without_format(
                {query.size(DIM_1), query.size(DIM_0), value.size(DIM_2)},
                query.options().dtype(query.dtype()));
        }
    }
    if (quant_scale_out.has_value()) {
        output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        if (query_rope.has_value()) {
            const at::Tensor &query_rope_tensor = c10::value_or_else(query_rope, [] { return at::Tensor(); });
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(query_rope_tensor.dtype()));
        } else {
            output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
        }
    } else {
        output = npu_preparation::apply_tensor_without_format(tmp_output);
    }

    at::Tensor softmax_lse;
    if (input_layout_str == "TND") {
        if (block_table.has_value()) { // IFA目前TND只支持PA场景，PFA目前TND只支持非PA场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_0), num_query_heads, 0},
                    c10::dtype(c10::ScalarType::Float));
            } else {
                softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_0), num_query_heads, 1},
                    c10::dtype(c10::ScalarType::Float));
            }
        } else {
            softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_0), query.size(DIM_1), 1},
                c10::dtype(c10::ScalarType::Float));
        }
    } else if (input_layout_str == "NTD_TND") {
        if (block_table.has_value()) { // pa场景
            if (query.size(DIM_2) == 0) { // 增加softmax lse的情况下，可能存在空tensor的分支
                softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_1), query.size(DIM_0), 0},
                    c10::dtype(c10::ScalarType::Float));
            } else {
                softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_1), query.size(DIM_0), 1},
                    c10::dtype(c10::ScalarType::Float));
            }
        } else {
            softmax_lse = npu_preparation::apply_tensor_without_format({query.size(DIM_1), query.size(DIM_0), 1},
                c10::dtype(c10::ScalarType::Float));
        }
    } else {
        softmax_lse = npu_preparation::apply_tensor_without_format({batchSize, num_query_heads, qsSize, 1},
            c10::dtype(c10::ScalarType::Float));
    }

    if (!return_softmax_lse) {
        softmax_lse = npu_preparation::apply_tensor_without_format({1}, c10::dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_v2_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor actual_shared_prefix_len {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
        quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor &, at::Tensor &> npu_fused_infer_attention_score_v2_out_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype,
    const c10::optional<at::Tensor> &workspace,
    at::Tensor &attention_out,
    at::Tensor &softmax_lse)
{
    // convert str
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor actual_shared_prefix_len {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    if (workspace.has_value()) {
        void* workspace_addr = const_cast<void *>(workspace.value().storage().data());
        uint64_t workspace_size = static_cast<uint64_t>(workspace.value().numel() * workspace.value().element_size());
        EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, workspace_addr, workspace_size, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, attention_out, softmax_lse);
    } else {
        EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
            quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
            dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, attention_out, softmax_lse);
    }
    return std::tuple<at::Tensor&, at::Tensor&>(attention_out, softmax_lse);
}

at::Tensor _npu_fused_infer_attention_score_v2_get_max_workspace_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_qlen,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_kvlen,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key,
    const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value,
    const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope,
    const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out,
    const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale,
    int64_t pre_tokens, int64_t next_tokens, c10::string_view input_layout,
    int64_t sparse_mode, int64_t block_size,
    int64_t query_quant_mode, int64_t key_quant_mode, int64_t value_quant_mode,
    int64_t inner_precise, bool return_softmax_lse,
    c10::optional<int64_t> query_dtype, c10::optional<int64_t> key_dtype, c10::optional<int64_t> value_dtype,
    c10::optional<int64_t> query_rope_dtype, c10::optional<int64_t> key_rope_dtype,
    c10::optional<int64_t> key_shared_prefix_dtype, c10::optional<int64_t> value_shared_prefix_dtype,
    c10::optional<int64_t> dequant_scale_query_dtype, c10::optional<int64_t> dequant_scale_key_dtype,
    c10::optional<int64_t> dequant_scale_value_dtype, c10::optional<int64_t> dequant_scale_key_rope_dtype)
{
    std::string input_layout_str = std::string(input_layout);

    // construct the output tensor
    std::tuple<at::Tensor, at::Tensor> fia_output = op_api::construct_fia_output_tensor_v2(query, value, input_layout_str,
                                                                                           quant_scale_out, block_table, num_query_heads,
                                                                                           num_key_value_heads,
                                                                                           return_softmax_lse, query_rope);
    at::Tensor output = std::get<0>(fia_output);
    at::Tensor softmax_lse = std::get<1>(fia_output);

    // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor actual_shared_prefix_len {nullptr};
    at::Tensor dequant_scale1;
    at::Tensor quant_scale1;
    at::Tensor dequant_scale2;
    at::Tensor antiquant_scale;
    at::Tensor antiquant_offset;
    at::Tensor query_padding_size;
    at::Tensor kv_padding_size;
    at::Tensor key_shared_prefix;
    at::Tensor value_shared_prefix;
    int64_t antiquant_mode = 0;

    at::TensorList valueTensors = value;
    at::TensorList keyTensors = key;

    uint64_t workspace_size = EXEC_GET_MAX_WORKSPACE_CMD(aclnnFusedInferAttentionScoreV4, query, keyTensors, valueTensors, pse_shift, atten_mask, actual_seq_qlen, actual_seq_kvlen, dequant_scale1, quant_scale1, dequant_scale2,
        quant_scale_out, quant_offset_out, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, dequant_scale_key, dequant_offset_key, dequant_scale_value,
        dequant_offset_value, key_shared_prefix, value_shared_prefix, actual_shared_prefix_len, query_rope, key_rope, dequant_scale_key_rope, dequant_scale_query, learnable_sink, num_query_heads, softmax_scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, return_softmax_lse, key_quant_mode, value_quant_mode, query_quant_mode, output, softmax_lse);
    at::Tensor workspace_tensor = npu_preparation::apply_tensor_without_format({workspace_size}, query.options().dtype(query.dtype()));
    return workspace_tensor;
}

#endif
} // namespace op_api
