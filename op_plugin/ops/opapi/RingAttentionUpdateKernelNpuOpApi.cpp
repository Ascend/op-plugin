// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
namespace {
constexpr int64_t SOFTMAX_LAST_DIM = 8;

bool is_supported_attn_dtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kFloat || dtype == at::kBFloat16;
}

void check_attention_inputs(
    const at::Tensor &prev_attn_out,
    const at::Tensor &prev_softmax_max,
    const at::Tensor &prev_softmax_sum,
    const at::Tensor &cur_attn_out,
    const at::Tensor &cur_softmax_max,
    const at::Tensor &cur_softmax_sum,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    c10::string_view input_layout,
    c10::string_view input_softmax_layout)
{
    std::string input_layout_str = std::string(input_layout);
    std::string input_softmax_layout_str = std::string(input_softmax_layout);

    TORCH_CHECK(
        input_layout_str == "SBH" || input_layout_str == "TND",
        "input_layout only supports 'SBH' or 'TND', but got ", input_layout_str,
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        input_softmax_layout_str.empty() || input_softmax_layout_str == "SBH" || input_softmax_layout_str == "TND",
        "input_softmax_layout only supports '', 'SBH' or 'TND', but got ", input_softmax_layout_str,
        OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(
        prev_attn_out.sizes() == cur_attn_out.sizes(),
        "prev_attn_out and cur_attn_out must have the same shape.",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        prev_attn_out.scalar_type() == cur_attn_out.scalar_type(),
        "prev_attn_out and cur_attn_out must have the same dtype.",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        is_supported_attn_dtype(prev_attn_out.scalar_type()),
        "attn_out dtype only supports float16, float32 or bfloat16, but got ",
        prev_attn_out.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));

    TORCH_CHECK(
        prev_softmax_max.sizes() == prev_softmax_sum.sizes() &&
        prev_softmax_max.sizes() == cur_softmax_max.sizes() &&
        prev_softmax_max.sizes() == cur_softmax_sum.sizes(),
        "softmax tensors must keep the same shape.",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        prev_softmax_max.scalar_type() == at::kFloat &&
        prev_softmax_sum.scalar_type() == at::kFloat &&
        cur_softmax_max.scalar_type() == at::kFloat &&
        cur_softmax_sum.scalar_type() == at::kFloat,
        "softmax tensors must use float32.",
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        prev_softmax_max.dim() > 0 && prev_softmax_max.size(prev_softmax_max.dim() - 1) == SOFTMAX_LAST_DIM,
        "softmax tensors require the last dimension to be 8.",
        OPS_ERROR(ErrCode::VALUE));

    TORCH_CHECK(
        prev_attn_out.dim() == 3,
        "prev_attn_out must be a 3D tensor, but got ", prev_attn_out.dim(), "D.",
        OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(
        cur_attn_out.dim() == 3,
        "cur_attn_out must be a 3D tensor, but got ", cur_attn_out.dim(), "D.",
        OPS_ERROR(ErrCode::VALUE));

    if (input_layout_str == "SBH") {
        TORCH_CHECK(
            prev_softmax_max.dim() == 4,
            "softmax tensors must be 4D when input_layout is 'SBH'.",
            OPS_ERROR(ErrCode::VALUE));
    } else {
        TORCH_CHECK(
            actual_seq_qlen.has_value() && actual_seq_qlen.value().defined(),
            "actual_seq_qlen is required when input_layout is 'TND'.",
            OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(
            prev_softmax_max.dim() == 3,
            "softmax tensors must be 3D when input_layout is 'TND'.",
            OPS_ERROR(ErrCode::VALUE));
    }

    if (actual_seq_qlen.has_value() && actual_seq_qlen.value().defined()) {
        TORCH_CHECK(
            actual_seq_qlen.value().scalar_type() == at::kLong,
            "actual_seq_qlen must use int64 dtype.",
            OPS_ERROR(ErrCode::TYPE));
    }
}
} // namespace

using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_ring_attention_update(
    const at::Tensor &prev_attn_out,
    const at::Tensor &prev_softmax_max,
    const at::Tensor &prev_softmax_sum,
    const at::Tensor &cur_attn_out,
    const at::Tensor &cur_softmax_max,
    const at::Tensor &cur_softmax_sum,
    const c10::optional<at::Tensor> &actual_seq_qlen,
    c10::string_view input_layout,
    c10::string_view input_softmax_layout)
{
    check_attention_inputs(
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
        actual_seq_qlen,
        input_layout,
        input_softmax_layout);

    const at::Tensor &actual_seq_qlen_tensor = actual_seq_qlen.value_or(at::Tensor());
    char *input_layout_ptr = const_cast<char *>(input_layout.data());
    char *input_softmax_layout_ptr = const_cast<char *>(input_softmax_layout.data());

    at::Tensor attn_out = npu_preparation::apply_tensor_without_format(prev_attn_out);
    at::Tensor softmax_max = npu_preparation::apply_tensor_without_format(prev_softmax_max);
    at::Tensor softmax_sum = npu_preparation::apply_tensor_without_format(prev_softmax_sum);

    EXEC_NPU_CMD(
        aclnnRingAttentionUpdateV2,
        prev_attn_out,
        prev_softmax_max,
        prev_softmax_sum,
        cur_attn_out,
        cur_softmax_max,
        cur_softmax_sum,
        actual_seq_qlen_tensor,
        input_layout_ptr,
        input_softmax_layout_ptr,
        attn_out,
        softmax_max,
        softmax_sum);

    return std::make_tuple(attn_out, softmax_max, softmax_sum);
}
} // namespace op_api
