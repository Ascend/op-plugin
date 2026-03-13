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
#include <string>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using namespace at_npu::native;
const int DIMENSION_4D = 4;
using npu_preparation = at_npu::native::OpPreparation;


// 入参检查
static void check_params(const at::Tensor &query,
                         const at::Tensor &key,
                         const at::Tensor &value,
                         int64_t inner_precise)
{
    // Q/K/V 数据类型必须一致
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
        "query, key, value must have the same dtype, got query=", query.scalar_type(),
        ", key=", key.scalar_type(), ", value=", value.scalar_type(), OPS_ERROR(ErrCode::PARAM));

    // 当Q为bf16时，inner_precise必须为0
    if (query.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(inner_precise == 0,
            "when query/key/value are bfloat16, inner_precise must be 0, got ", inner_precise,
            OPS_ERROR(ErrCode::PARAM));
    }
}

// PTA 接口实现
std::tuple<at::Tensor, at::Tensor> npu_block_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_sparse_mask,
    const c10::IntArrayRef block_shape,
    c10::string_view q_input_layout,
    c10::string_view kv_input_layout,
    int64_t num_key_value_heads,
    double scale_value,
    int64_t inner_precise,
    const c10::OptionalIntArrayRef actual_seq_lengths,
    const c10::OptionalIntArrayRef actual_seq_lengths_kv,
    c10::optional<int64_t> softmax_lse_flag)
{
    check_params(query, key, value, inner_precise);

    // 分配输出 Tensor
    auto attention_out_shape = query.sizes().vec();
    attention_out_shape[attention_out_shape.size() - 1] = value.size(-1);
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(attention_out_shape, query.options());

    at::Tensor softmax_lse_out;
    if (query.dim() == DIMENSION_4D) {
        softmax_lse_out = npu_preparation::apply_tensor_without_format(
            {query.size(0), query.size(1), query.size(2), 1}, query.options().dtype(at::kFloat));
    } else {
        softmax_lse_out = npu_preparation::apply_tensor_without_format(
            {query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
    }

    // 获取参数
    const at::IntArrayRef block_shape_value = block_shape;
    const at::IntArrayRef actual_seq_lengths_q_value = actual_seq_lengths.value_or(at::IntArrayRef{});
    const at::IntArrayRef actual_seq_lengths_kv_value = actual_seq_lengths_kv.value_or(at::IntArrayRef{});
    const int64_t softmax_lse_flag_value = softmax_lse_flag.value_or(0);

    // 初始化aclnn中的暂不支持参数
    const at::Tensor atten_mask = at::Tensor();
    const at::Tensor block_table = at::Tensor();
    const int64_t mask_type = 0;
    const int64_t block_size = 0;
    const int64_t pre_tokens = 2147483647;
    const int64_t next_tokens = 2147483647;

    // 获取到layout的指针，直接传给aclnn接口，供其获取字符串
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // 调用aclnn接口
    EXEC_NPU_NO_FORMAT_CHECK_CMD(
        aclnnBlockSparseAttention, query, key, value, block_sparse_mask, atten_mask,
        block_shape_value, actual_seq_lengths_q_value, actual_seq_lengths_kv_value, block_table,
        q_input_layout_ptr, kv_input_layout_ptr, num_key_value_heads, mask_type, scale_value,
        inner_precise, block_size, pre_tokens, next_tokens, softmax_lse_flag_value,
        attention_out, softmax_lse_out);

    // 返回结果
    return std::make_tuple(attention_out, softmax_lse_out);
}
}
