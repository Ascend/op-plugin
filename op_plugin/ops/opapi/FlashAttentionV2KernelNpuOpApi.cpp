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

#include <cstring>

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const int THIRD_ELEMENT = 2;
const int FORTH_ELEMENT = 3;
const int DIMENSION_3D = 3;
const int DIMENSION_4D = 4;
const int LAYOUT_MAX_LENGTH = 20;
const double EPSILON = 1e-9;
const int64_t LENGTH_BIAS = 32;
const static int64_t SOFTMAXMAX_LAST_DIMSHAPE = 8;
const static int64_t MAX_SEQUENCE_LENGTH = 1000000;
using namespace at_npu::native;
using npu_preparation = at_npu::native::OpPreparation;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

enum class SparseMode {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    PREFIX,
    PREFIX_COMPRESS,
    RIGHT_DOWN_CAUSAL_BAND,
    BAND_LEFT_UP_CAUSAL
};

namespace {
DropOutStatus get_dropout_status(double keep_prob)
{
    if (std::abs(keep_prob - 0.0) < EPSILON) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (std::abs(keep_prob - 1.0) < EPSILON) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
            "Expected all tensors to be on the same device. "
            "Expected NPU tensor, please check whether the input tensor device is correct.",
            OPS_ERROR(ErrCode::TYPE));
        return custom_ops::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

at::Tensor& stateless_dropout_gen_mask_aclop(const at::Tensor &query, double keep_prob, int64_t seed,
    const int64_t offset, const int64_t numels, at::Tensor& mask)
{
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    c10::TensorOptions options = query.options();
    at::SmallVector<int64_t, ::N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(at::Scalar(keep_prob), query.scalar_type(),  at_npu::native::CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(at::Scalar(seed), at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong,  at_npu::native::CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

at::Tensor dropout_gen_mask_impl(const at::Tensor &query, double keep_prob, int64_t seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    c10::TensorOptions options = query.options();
    at::Tensor mask = OpPreparation::apply_tensor_without_format(at::IntArrayRef{length}, options.dtype(at::kByte));
    c10::SmallVector<int64_t, SIZE> shapeSize = {numels};
    at::IntArrayRef shapeArray = at::IntArrayRef(shapeSize);
    double prob;
    at::Scalar probScalar;
    if (query.scalar_type() == at::kHalf) {
        probScalar = at::Scalar(at::Half(1.0)- at::Half(keep_prob));
    } else if (query.scalar_type() == at::kBFloat16) {
        probScalar = at::Scalar(at::BFloat16(1.0)- at::BFloat16(keep_prob));
    } else {
        probScalar = at::Scalar(float(1.0) - float(keep_prob));
    }
    prob = probScalar.toDouble();
    aclDataType probDataType = at_npu::native::OpPreparation::convert_to_acl_data_type(query.scalar_type());
    DO_COMPATIBILITY(aclnnDropoutGenMaskV2,
                     stateless_dropout_gen_mask_aclop(query, keep_prob, seed, offset, numels, mask));
    EXEC_NPU_CMD(aclnnDropoutGenMaskV2, shapeArray, prob, seed, offset, probDataType, mask);
    return mask;
}

at::Tensor dropout_gen_mask_dispatch(const at::Tensor &query, double keep_prob, int64_t seed,
    const int64_t offset, const int64_t numels, const bool gen_mask_parallel, const bool sync)
{
    at::Tensor mask;

    if (gen_mask_parallel) {
        auto original_stream = c10_npu::getCurrentNPUStream();
        {
            // During the life cycle of this raii instance, the calcu stream is set as the
            // secondary stream, and tasks are distributed to the secondary stream. At the
            // same time, according to the one-stream-one-pool principle, memory is also
            // alloced from the pool of the secondary stream.
            c10_npu::SecondaryStreamGuard guard(c10_npu::getCurrentSecondaryStream());
            mask = dropout_gen_mask_impl(query, keep_prob, seed, offset, numels);
            if (sync) {
                OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        }
    } else {
        mask = dropout_gen_mask_impl(query, keep_prob, seed, offset, numels);
    }
    return mask;
}
} // namespace _

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor static dropout_gen_mask(const at::Tensor &query, const at::Tensor &key, double keep_prob, int64_t head_num,
    std::string input_layout, bool gen_mask_parallel, bool sync, int64_t &seed, int64_t &offset, int64_t &numels)
{
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = query.size(0) * head_num * query.size(1) * key.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = query.size(1) * head_num * query.size(0) * key.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = query.size(0) * query.size(1) * query.size(THIRD_ELEMENT) * key.size(THIRD_ELEMENT); // [B,N,S,S]
    } else if (input_layout == "BSND") {
        numels = query.size(0) * query.size(THIRD_ELEMENT) * query.size(1) * key.size(1); // [B,N,S,S]
    }
    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += LENGTH_BIAS;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = static_cast<int64_t>(pair.first);
        offset = static_cast<int64_t>(pair.second);
        drop_mask = dropout_gen_mask_dispatch(query, keep_prob, seed, offset, numels, gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }
    return drop_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_backward_v2(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &dy, int64_t head_num, const std::string input_layout,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &drop_mask,
    const c10::optional<at::Tensor> &padding_mask, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max, const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in, const c10::optional<at::Tensor> &attention_in,
    const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope, double scale_value,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise, c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen, c10::OptionalIntArrayRef actual_seq_kvlen,
    c10::OptionalIntArrayRef q_start_idx, c10::OptionalIntArrayRef kv_start_idx, int64_t sparse_mode, int64_t pse_type)
{
    double scale = scale_value;

    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &softmax_max_const = softmax_max.value_or(at::Tensor());
    const at::Tensor &softmax_sum_const = softmax_sum.value_or(at::Tensor());
    const at::Tensor &softmax_const = softmax_in.value_or(at::Tensor());
    const at::Tensor &attention_const = attention_in.value_or(at::Tensor());
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    auto prefixN = prefix.value_or(at::IntArrayRef{});
    auto ac_seq_qlen = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto ac_seq_kvlen = actual_seq_kvlen.value_or(at::IntArrayRef{});
    auto q_start_idx_val = q_start_idx.value_or(at::IntArrayRef{});
    auto kv_start_idx_val = kv_start_idx.value_or(at::IntArrayRef{});

    at::Tensor format_query = format_trans(query);
    at::Tensor format_query_rope = format_trans(query_rope_const);
    at::Tensor format_key = format_trans(key);
    at::Tensor format_key_rope = format_trans(key_rope_const);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_dy = format_trans(dy);

    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_drop_mask = format_trans(drop_mask_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_softmax_max = format_trans(softmax_max_const);
    at::Tensor format_softmax_sum = format_trans(softmax_sum_const);
    at::Tensor format_softmax = format_trans(softmax_const);
    at::Tensor format_attention = format_trans(attention_const);
    at::Tensor dq = OpPreparation::apply_tensor_without_format(format_query);
    at::Tensor dk = OpPreparation::apply_tensor_without_format(format_key);
    at::Tensor dv = OpPreparation::apply_tensor_without_format(format_value);
    at::Tensor dpse;
    at::Tensor dq_rope;
    at::Tensor dk_rope;
    if (format_pse.defined()) {
        dpse = OpPreparation::apply_tensor_without_format(format_pse);
    } else {
        dpse = at::empty({0}, query.options());
    }
    if (format_query_rope.defined()) {
        dq_rope = OpPreparation::apply_tensor_without_format(format_query_rope);
    } else {
        dq_rope = at::empty({0}, query.options());
    }
    if (format_key_rope.defined()) {
        dk_rope = OpPreparation::apply_tensor_without_format(format_key_rope);
    } else {
        dk_rope = at::empty({0}, key.options());
    }

    char input_layout_char[LAYOUT_MAX_LENGTH];
    strncpy(input_layout_char, input_layout.c_str(), LAYOUT_MAX_LENGTH - 1);
    if (format_query_rope.defined() && format_key_rope.defined()) {
        if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
            EXEC_NPU_CMD(
                aclnnFlashAttentionUnpaddingScoreGradV3, format_query, format_query_rope, format_key, format_key_rope, format_value, format_dy,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                format_softmax_sum, format_softmax, format_attention, prefixN, ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val,
                scale_value, keep_prob, pre_tokens, next_tokens, head_num, input_layout_char, inner_precise, sparse_mode, pse_type,
                dq, dq_rope, dk, dk_rope, dv, dpse);
        }
    } else {
        if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
            EXEC_NPU_CMD(
                aclnnFlashAttentionUnpaddingScoreGradV2, format_query, format_key, format_value, format_dy,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                format_softmax_sum, format_softmax, format_attention, prefixN, ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val,
                scale_value, keep_prob, pre_tokens, next_tokens, head_num, input_layout_char, inner_precise, sparse_mode, pse_type,
                dq, dk, dv, dpse);
        } else {
            EXEC_NPU_CMD(
                aclnnFlashAttentionScoreGradV2, format_query, format_key, format_value, format_dy,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, format_softmax_max,
                format_softmax_sum, format_softmax, format_attention, prefixN, q_start_idx_val, kv_start_idx_val, scale_value, keep_prob,
                pre_tokens, next_tokens, head_num, input_layout_char, inner_precise, sparse_mode, pse_type, dq, dk, dv, dpse);
        }
    }

    FLOP_COUNT(FlopCounter::flash_attention_backward_flop, query, key, value,
        dy, head_num, input_layout, actual_seq_qlen, actual_seq_kvlen);

    if (!format_pse.defined()) {
        at::Tensor dpse_required;
        dpse = dpse_required;
    }
    if (!format_query_rope.defined()) {
        at::Tensor dq_rope_required;
        dq_rope = dq_rope_required;
    }
    if (!format_key_rope.defined()) {
        at::Tensor dk_rope_required;
        dk_rope = dk_rope_required;
    }

    return std::make_tuple(dq, dk, dv, dpse, dq_rope, dk_rope);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &dy,
    int64_t head_num,
    c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &softmax_max,
    const c10::optional<at::Tensor> &softmax_sum,
    const c10::optional<at::Tensor> &softmax_in,
    const c10::optional<at::Tensor> &attention_in,
    const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope,
    double scale_value,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t inner_precise,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    c10::OptionalIntArrayRef prefix,
    c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen,
    int64_t sparse_mode,
    bool gen_mask_parallel,
    bool sync,
    int64_t pse_type,
    c10::OptionalIntArrayRef q_start_idx,
    c10::OptionalIntArrayRef kv_start_idx)
{
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ",
        value.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dy.dim() == DIMENSION_3D || dy.dim() == DIMENSION_4D,
        "The shapes of the input dy should be 3 or 4 dimensional, but got ", dy.dim(), "-dimensional",
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(keep_prob > 0 && keep_prob <= 1,
        "The keep_prob value must be in range of (0, 1], but got ", keep_prob,
        OPS_ERROR(ErrCode::PARAM));
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD" ||
        input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ", input_layout, OPS_ERROR(ErrCode::PARAM));

    int64_t length = (numels + 128 - 1) / 128 * 128 / 8;
    length += LENGTH_BIAS;
    at::Tensor drop_mask;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = dropout_gen_mask_dispatch(query, keep_prob, seed, offset, numels, gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }
    auto result = npu_fusion_attention_backward_v2(query,
        key, value, dy, head_num, input_layout_str, pse, drop_mask, padding_mask, atten_mask,
        softmax_max, softmax_sum, softmax_in, attention_in, query_rope, key_rope, scale_value, keep_prob, pre_tokens,
        next_tokens, inner_precise, prefix, actual_seq_qlen, actual_seq_kvlen, q_start_idx, kv_start_idx, sparse_mode, pse_type);
    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_fusion_attention_v2(
    const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor> &pse, const c10::optional<at::Tensor> &padding_mask,
    const c10::optional<at::Tensor> &atten_mask, const c10::optional<at::Tensor> &query_rope, const c10::optional<at::Tensor> &key_rope,
    double scale, double keep_prob, int64_t pre_tokens, int64_t next_tokens, int64_t inner_precise,
    c10::OptionalIntArrayRef prefix, c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen, int64_t sparse_mode, bool gen_mask_parallel, bool sync,
    int64_t pse_type, c10::OptionalIntArrayRef q_start_idx, c10::OptionalIntArrayRef kv_start_idx)
{
    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &padding_mask_const = padding_mask.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &query_rope_const = query_rope.value_or(at::Tensor());
    const at::Tensor &key_rope_const = key_rope.value_or(at::Tensor());
    auto prefixN = prefix.value_or(at::IntArrayRef{});
    auto ac_seq_qlen = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto ac_seq_kvlen = actual_seq_kvlen.value_or(at::IntArrayRef{});
    auto q_start_idx_val = q_start_idx.value_or(at::IntArrayRef{});
    auto kv_start_idx_val = kv_start_idx.value_or(at::IntArrayRef{});

    TORCH_CHECK(head_num > 0, "head_num must > 0, but got ", head_num, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(query.dim() == DIMENSION_3D || query.dim() == DIMENSION_4D,
        "The shapes of the input query should be 3 or 4 dimensional, but got ",
        query.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (query_rope_const.defined()) {
        TORCH_CHECK(query_rope_const.dim() == DIMENSION_3D || query_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input query_rope should be 3 or 4 dimensional, but got ",
            query_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(key.dim() == DIMENSION_3D || key.dim() == DIMENSION_4D,
        "The shapes of the input key should be 3 or 4 dimensional, but got ", key.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    if (key_rope_const.defined()) {
        TORCH_CHECK(key_rope_const.dim() == DIMENSION_3D || key_rope_const.dim() == DIMENSION_4D,
            "The shapes of the input key_rope should be 3 or 4 dimensional, but got ",
            key_rope_const.dim(), "-dimensional", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(ac_seq_qlen.size() != 0 && ac_seq_kvlen.size() != 0,
            "the size of actual_seq_qlen and actual_seq_kvlen cannot be empty." + OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(value.dim() == DIMENSION_3D || value.dim() == DIMENSION_4D,
        "The shapes of the input value should be 3 or 4 dimensional, but got ", value.dim(),
        "-dimensional", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(keep_prob > 0 && keep_prob <= 1,
        "The keep_prob value must be in range of (0, 1], but got ", keep_prob, OPS_ERROR(ErrCode::PARAM));
    std::string input_layout_str = std::string(input_layout);
    if (input_layout_str == "TND") {
        TORCH_CHECK((sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode < static_cast<int64_t>(SparseMode::PREFIX)) ||
                    (sparse_mode > static_cast<int64_t>(SparseMode::PREFIX) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)),
                    "The sparse_mode value must be in range of [0,5) or (5,8], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(ac_seq_qlen.size() != 0 && ac_seq_kvlen.size() != 0 && ac_seq_qlen.size() == ac_seq_kvlen.size(),
                    "the size of actual_seq_qlen and actual_seq_kvlen must be the same and cannot be empty." +
                    OPS_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK(sparse_mode >= static_cast<int64_t>(SparseMode::NO_MASK) &&
                    sparse_mode <= static_cast<int64_t>(SparseMode::PREFIX_COMPRESS),
                    "The sparse_mode value must be in range of [0,6], but got ",
                    sparse_mode, OPS_ERROR(ErrCode::PARAM));
    }
    for (auto &c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" ||
        input_layout_str == "BNSD" || input_layout_str == "BSND" || input_layout_str == "TND",
        "The input_layout should be BSH/SBH/BNSD/BSND/TND(case-insensitive), but got ",
        input_layout, OPS_ERROR(ErrCode::PARAM));

    int64_t B = 0;
    int64_t S0 = 0; // S for query
    int64_t S1 = 0; // S for key & value
    int64_t N_local = 0; // N for npu_fusion_attention
    int64_t D = 0;
    int64_t H = 0;
    int64_t T = 0;
    int64_t D2 = 0; // D2 for value head-dim
    c10::SmallVector<int64_t> atten_score_shape;

    if (input_layout_str == "BSH") {
        B = query.size(0);
        S0 = query.size(1);
        S1 = key.size(1);
        H = query.size(THIRD_ELEMENT);
        D = H / head_num;
        D2 = (D == 0 || !key.size(THIRD_ELEMENT)) ? 0 : value.size(THIRD_ELEMENT) / (key.size(THIRD_ELEMENT) / D);
        atten_score_shape = {B, S0, head_num * D2};
    } else if (input_layout_str == "SBH") {
        B = query.size(1);
        S0 = query.size(0);
        S1 = key.size(0);
        H = query.size(THIRD_ELEMENT);
        D = H / head_num;
        D2 = (D == 0 || !key.size(THIRD_ELEMENT)) ? 0 : value.size(THIRD_ELEMENT) / (key.size(THIRD_ELEMENT) / D);
        atten_score_shape = {S0, B, head_num * D2};
    } else if (input_layout_str == "BNSD") {
        B = query.size(0);
        N_local = query.size(1);
        S0 = query.size(THIRD_ELEMENT);
        S1 = key.size(THIRD_ELEMENT);
        D = query.size(FORTH_ELEMENT);
        D2 = value.size(FORTH_ELEMENT);
        atten_score_shape = {B, N_local, S0, D2};
    } else if (input_layout_str == "BSND") {
        B = query.size(0);
        N_local = query.size(THIRD_ELEMENT);
        S0 = query.size(1);
        S1 = key.size(1);
        D = query.size(FORTH_ELEMENT);
        D2 = value.size(FORTH_ELEMENT);
        atten_score_shape = {B, S0, N_local, D2};
    } else if (input_layout_str == "TND") {
        T = query.size(0);
        N_local = query.size(1);
        D = query.size(THIRD_ELEMENT);
        D2 = value.size(THIRD_ELEMENT);
        atten_score_shape = {T, N_local, D2};
    }

    double scale_value = scale;

    at::Tensor format_query = format_trans(query);
    at::Tensor format_query_rope = format_trans(query_rope_const);
    at::Tensor attention_score = npu_preparation::apply_tensor_without_format(atten_score_shape, query.options());
    at::Tensor format_key = format_trans(key);
    at::Tensor format_key_rope = format_trans(key_rope_const);
    at::Tensor format_value = format_trans(value);
    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_padding_mask = format_trans(padding_mask_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);

    int64_t seed;
    int64_t offset;
    int64_t numels;
    for (size_t i = 0; i < ac_seq_qlen.size(); i++) {
        TORCH_CHECK(ac_seq_qlen[i] <= MAX_SEQUENCE_LENGTH && ac_seq_kvlen[i] <= MAX_SEQUENCE_LENGTH,
            "The sequence length should not greater than 1M, but got q", ac_seq_qlen[i], "kv", ac_seq_kvlen[i]);
    }

    if (input_layout_str == "TND" && ac_seq_qlen.size() == ac_seq_kvlen.size()) {
        numels = N_local;
        int64_t accum = ac_seq_qlen[0] * ac_seq_kvlen[0];
        for (size_t i = 1; i < ac_seq_qlen.size(); i++) {
            accum += ((ac_seq_qlen[i] - ac_seq_qlen[i - 1]) * (ac_seq_kvlen[i] - ac_seq_kvlen[i - 1]));
        }
        numels *= accum;
    }
    at::Tensor format_drop_mask = dropout_gen_mask(format_query, format_key, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::Tensor softmax_out;

    if (input_layout_str != "TND") {
        softmax_max = OpPreparation::apply_tensor_without_format({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [B, N, S0, 8]
        softmax_sum = OpPreparation::apply_tensor_without_format({B, head_num, S0, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [B, N, S0, 8]
    } else {
        softmax_max = OpPreparation::apply_tensor_without_format({T, N_local, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [T, N, 8]
        softmax_sum = OpPreparation::apply_tensor_without_format({T, N_local, SOFTMAXMAX_LAST_DIMSHAPE},
            query.options().dtype(at::kFloat)); // [T, N, 8]
    }
    softmax_out = at::empty({0}, query.options());
    char input_layout_char[LAYOUT_MAX_LENGTH];
    strncpy(input_layout_char, input_layout_str.c_str(), LAYOUT_MAX_LENGTH - 1);
    if (format_query_rope.defined() && format_key_rope.defined()) {
        if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
            EXEC_NPU_CMD(
                aclnnFlashAttentionVarLenScoreV3, format_query, format_query_rope, format_key, format_key_rope, format_value,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
                ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val, scale, keep_prob, pre_tokens, next_tokens, head_num,
                input_layout_char, inner_precise, sparse_mode, pse_type, softmax_max, softmax_sum,
                softmax_out, attention_score);
        }
    } else {
        if (!ac_seq_qlen.empty() && !ac_seq_kvlen.empty()) {
            EXEC_NPU_CMD(
                aclnnFlashAttentionVarLenScoreV2, format_query, format_key, format_value,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN,
                ac_seq_qlen, ac_seq_kvlen, q_start_idx_val, kv_start_idx_val, scale, keep_prob, pre_tokens, next_tokens, head_num,
                input_layout_char, inner_precise, sparse_mode, pse_type, softmax_max, softmax_sum,
                softmax_out, attention_score);
        } else {
            EXEC_NPU_CMD(
                aclnnFlashAttentionScoreV2, format_query, format_key, format_value,
                format_pse, format_drop_mask, format_padding_mask, format_atten_mask, prefixN, q_start_idx_val, kv_start_idx_val,
                scale, keep_prob, pre_tokens, next_tokens, head_num, input_layout_char, inner_precise,
                sparse_mode, pse_type, softmax_max, softmax_sum, softmax_out, attention_score);
        }
    }

    FLOP_COUNT(FlopCounter::flash_attention_forward_flop, query, key, value, head_num,
               input_layout_str, actual_seq_qlen, actual_seq_kvlen);

    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    return std::make_tuple(attention_score, softmax_max, softmax_sum, softmax_out,
        seed, offset, numels);
}
#endif
}
