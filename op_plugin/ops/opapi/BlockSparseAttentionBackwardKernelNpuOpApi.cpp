// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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
const int64_t MAX_HEAD_DIM = 128;
using npu_preparation = at_npu::native::OpPreparation;


// Validate input parameters.
static void check_params(const at::Tensor &query,
                         const at::Tensor &key,
                         const at::Tensor &value,
                         const c10::OptionalIntArrayRef actual_seq_lengths,
                         const c10::OptionalIntArrayRef actual_seq_lengths_kv,
                         c10::string_view q_input_layout,
                         c10::string_view kv_input_layout)
{
    // Q/K/V must use the same dtype.
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
        "query, key, value must have the same dtype, got query=", query.scalar_type(),
        ", key=", key.scalar_type(), ", value=", value.scalar_type(), OPS_ERROR(ErrCode::PARAM));

    // The kernel supports head_dim up to 128.
    int64_t head_dim = query.size(-1);
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM,
        "head_dim must be <= ", MAX_HEAD_DIM, ", but got ", head_dim, OPS_ERROR(ErrCode::PARAM));

    // TND inputs require non-empty per-batch actual sequence lengths.
    if (q_input_layout == "TND") {
        TORCH_CHECK(actual_seq_lengths.has_value() && actual_seq_lengths->size() > 0,
            "actual_seq_lengths must be specified when q_input_layout is TND",
            OPS_ERROR(ErrCode::PARAM));
    }
    if (kv_input_layout == "TND") {
        TORCH_CHECK(actual_seq_lengths_kv.has_value() && actual_seq_lengths_kv->size() > 0,
            "actual_seq_lengths_kv must be specified when kv_input_layout is TND",
            OPS_ERROR(ErrCode::PARAM));
    }
}


// PTA API implementation.
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_block_sparse_attention_backward(
    const at::Tensor &d_out,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &attention_out,
    const at::Tensor &softmax_lse,
    const at::Tensor &block_sparse_mask,
    const c10::OptionalIntArrayRef block_shape,
    const c10::OptionalIntArrayRef actual_seq_lengths,
    const c10::OptionalIntArrayRef actual_seq_lengths_kv,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value)
{
    check_params(query, key, value, actual_seq_lengths, actual_seq_lengths_kv, q_input_layout, kv_input_layout);

    at::Tensor d_query = npu_preparation::apply_tensor_without_format(query);
    at::Tensor d_key = npu_preparation::apply_tensor_without_format(key);
    at::Tensor d_value = npu_preparation::apply_tensor_without_format(value);

    // Use the default block shape [128, 128] when block_shape is not specified.
    static const int64_t kDefaultBlockShape[2] = {128, 128};
    const at::IntArrayRef block_shape_value = (block_shape.has_value() && block_shape->size() >= 2)
        ? *block_shape
        : at::IntArrayRef(kDefaultBlockShape, 2);

    // Initialize aclnn parameters that are not exposed by this PTA API.
    const at::Tensor atten_mask{nullptr};
    const int64_t mask_type = 0;
    const int64_t pre_tokens = 2147483647;
    const int64_t next_tokens = 2147483647;

    // Pass layout strings through to aclnn. aclnn owns the final validation of supported layouts.
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // Call aclnn API.
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnBlockSparseAttentionGrad,
        d_out, query, key, value,
        attention_out, softmax_lse,
        block_sparse_mask, atten_mask, block_shape_value,
        actual_seq_lengths, actual_seq_lengths_kv,
        q_input_layout_ptr, kv_input_layout_ptr,
        num_key_value_heads, mask_type, scale_value,
        pre_tokens, next_tokens,
        d_query, d_key, d_value);

    // Return gradients.
    return std::make_tuple(d_query, d_key, d_value);
}
}
