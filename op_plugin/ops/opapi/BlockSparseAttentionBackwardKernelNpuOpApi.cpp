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


// 入参检查
static void check_params(const at::Tensor &query,
                         const at::Tensor &key,
                         const at::Tensor &value)
{
    // Q/K/V 数据类型必须一致
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
        "query, key, value must have the same dtype, got query=", query.scalar_type(),
        ", key=", key.scalar_type(), ", value=", value.scalar_type(), OPS_ERROR(ErrCode::PARAM));

    // head_dim 不能超过 128
    int64_t head_dim = query.size(-1);
    TORCH_CHECK(head_dim <= MAX_HEAD_DIM,
        "head_dim must be <= ", MAX_HEAD_DIM, ", but got ", head_dim, OPS_ERROR(ErrCode::PARAM));
}


// PTA 接口实现
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
    check_params(query, key, value);

    // 分配输出 Tensor
    at::Tensor d_query = npu_preparation::apply_tensor_without_format(query);
    at::Tensor d_key = npu_preparation::apply_tensor_without_format(key);
    at::Tensor d_value = npu_preparation::apply_tensor_without_format(value);

    // blockShape 非空，未传时使用默认 [128, 128]
    static const int64_t kDefaultBlockShape[2] = {128, 128};
    const at::IntArrayRef block_shape_value = (block_shape.has_value() && block_shape->size() >= 2)
        ? *block_shape
        : at::IntArrayRef(kDefaultBlockShape, 2);

    // 初始化 aclnn 中的暂不支持参数
    const at::Tensor atten_mask{nullptr};
    const int64_t mask_type = 0;
    const int64_t pre_tokens = 2147483647;
    const int64_t next_tokens = 2147483647;

    // 获取到 layout 的指针，直接传给 aclnn 接口，供其获取字符串
    char *q_input_layout_ptr = const_cast<char *>(q_input_layout.data());
    char *kv_input_layout_ptr = const_cast<char *>(kv_input_layout.data());

    // 调用alcnn接口
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

    // 返回结果
    return std::make_tuple(d_query, d_key, d_value);
}
}
