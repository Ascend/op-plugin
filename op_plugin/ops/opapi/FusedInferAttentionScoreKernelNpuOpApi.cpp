// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
const static int FLASH_THRESHOLD = 512;
const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_fused_infer_attention_score_symint(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths,
    c10::OptionalArrayRef<c10::SymInt> actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &dequant_scale1,
    const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2,
    const c10::optional<at::Tensor> &quant_scale2,
    const c10::optional<at::Tensor> &quant_offset2,
    const c10::optional<at::Tensor> &antiquant_scale,
    const c10::optional<at::Tensor> &antiquant_offset,
    const c10::optional<at::Tensor> &key_antiquant_scale,
    const c10::optional<at::Tensor> &key_antiquant_offset,
    const c10::optional<at::Tensor> &value_antiquant_scale,
    const c10::optional<at::Tensor> &value_antiquant_offset,
    const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &query_padding_size,
    const c10::optional<at::Tensor> &kv_padding_size,
    const c10::optional<at::Tensor> &key_shared_prefix,
    const c10::optional<at::Tensor> &value_shared_prefix,
    c10::OptionalArrayRef<c10::SymInt> actual_shared_prefix_len,
    int64_t num_heads, double scale,
    int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise,
    int64_t block_size, int64_t antiquant_mode,
    int64_t key_antiquant_mode, int64_t value_antiquant_mode,
    bool softmax_lse_flag)
{
    // construct the output tensor
    at::Tensor output;
    int64_t batchSize = 1;
    int64_t qsSize = 1;
    at::Tensor tmp_output = npu_preparation::apply_tensor_without_format(query);
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "BNSD_BSND") {
        tmp_output = OpPreparation::apply_tensor_without_format({query.size(0), query.size(2), query.size(1), query.size(3)},
            query.options().dtype(query.dtype()));
        batchSize = query.size(0);
        qsSize = query.size(2);
    } else if (input_layout_str == "NSD") {
        batchSize = 1;
        qsSize = query.size(1);
    } else if (input_layout_str == "BSH") {
        batchSize = query.size(0);
        qsSize = query.size(1);
    } else if (input_layout_str == "BSND") {
        batchSize = query.size(0);
        qsSize = query.size(1);
    } else if (input_layout_str == "BNSD") {
        batchSize = query.size(0);
        qsSize = query.size(2);
    }
    if (quant_scale2.has_value()) {
        output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Char));
    } else if (query.dtype() == at::kChar) {
        output = npu_preparation::apply_tensor_without_format(tmp_output.sizes(), c10::dtype(c10::ScalarType::Half));
    } else {
        output = npu_preparation::apply_tensor_without_format(tmp_output);
    }

    // convert str
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::TensorList keyTensors = key;
    at::TensorList valueTensors = value;
    auto actSeqLenMiddle = actual_seq_lengths.value_or(at::ArrayRef<c10::SymInt>{});
    auto actSeqLen = c10::asIntArrayRefUnchecked(actSeqLenMiddle);
    auto actSeqLenKvMiddle = actual_seq_lengths_kv.value_or(at::ArrayRef<c10::SymInt>{});
    auto actSeqLenKv = c10::asIntArrayRefUnchecked(actSeqLenKvMiddle);
    auto actSeqLenPrefixMiddle = actual_shared_prefix_len.value_or(at::ArrayRef<c10::SymInt>{});
    auto actSeqLenPrefix = c10::asIntArrayRefUnchecked(actSeqLenPrefixMiddle);

    // construct softmax_lse tensor
    at::Tensor softmax_lse = npu_preparation::apply_tensor_without_format(
        {batchSize, num_heads, qsSize, 1}, c10::dtype(c10::ScalarType::Float));

    if (softmax_lse_flag != true) {
        softmax_lse = npu_preparation::apply_tensor_without_format({1}, c10::dtype(c10::ScalarType::Float));
    }
    // dispatch hostAPI
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnFusedInferAttentionScoreV2, query, keyTensors, valueTensors, pse_shift, atten_mask, actSeqLen, actSeqLenKv, dequant_scale1, quant_scale1, dequant_scale2,
        quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale,
        value_antiquant_offset, key_shared_prefix, value_shared_prefix, actSeqLenPrefix, num_heads, scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode, output, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}
#endif
} // namespace op_api
