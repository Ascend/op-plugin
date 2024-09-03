// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
using namespace at_npu::native;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

namespace {
DropOutStatus get_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor tensor_format_trans(const at::Tensor &at_tensor)
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

at::Tensor gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    c10::TensorOptions options = self.options();
    at::Tensor mask = OpPreparation::apply_tensor_without_format(at::IntArrayRef{length + 32}, options.dtype(at::kByte));
    at::SmallVector<int64_t, ::N> offsetList = {0, offset};
    const int64_t seed1 = 0;
    OpCommand cmd;
    cmd.Name("StatelessDropOutGenMask")
        .Input(at::IntArrayRef{numels})
        .Input(keep_prob, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(seed, at::ScalarType::Int)
        .Input(at::Scalar(seed1), at::ScalarType::Int)
        .Input(offsetList, at::kLong, CompileType::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Output(mask)
        .Run();
    return mask;
}

at::Tensor gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
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
            mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
            if (sync) {
                OPS_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        }
    } else {
        mask = gen_mask_impl(self, keep_prob, seed, offset, numels);
    }
    return mask;
}

at::Tensor gen_mask(const at::Tensor &self, double keep_prob,
    int64_t head_num, std::string input_layout, bool gen_mask_parallel, bool sync,
    int64_t &seed, int64_t &offset, int64_t &numels)
{
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
    } else if (input_layout == "BNSD") {
        numels = self.size(0) * self.size(1) * self.size(2) * self.size(2); // [B,N,S,S]
    }
    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = static_cast<int64_t>(pair.first);
        offset = static_cast<int64_t>(pair.second);
        drop_mask = gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed),
            offset, numels, gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }
    return drop_mask;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_multi_head_attention_v2_backward(
    const at::Tensor &attention_score_grad,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &softmax_log_max_sum,
    const at::Tensor &attention_score,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &alibi_mask,
    const c10::optional<at::Tensor> &drop_mask,
    double scale_value,
    int64_t head_num,
    const std::string input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool gen_mask_parallel,
    bool sync)
{
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    at::Tensor format_drop_mask = tensor_format_trans(drop_mask_const);

    at::Tensor format_attention_score_grad = tensor_format_trans(attention_score_grad);
    at::Tensor format_query = tensor_format_trans(query);
    at::Tensor format_key = tensor_format_trans(key);
    at::Tensor format_value = tensor_format_trans(value);
    at::Tensor format_softmax_log_max_sum = tensor_format_trans(softmax_log_max_sum);
    at::Tensor format_attention_score = tensor_format_trans(attention_score);

    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    at::Tensor format_atten_mask = tensor_format_trans(atten_mask_const);
    const at::Tensor &alibi_mask_const = alibi_mask.value_or(at::Tensor());
    at::Tensor format_alibi_mask = tensor_format_trans(alibi_mask_const);

    at::Tensor query_grad;
    at::Tensor key_grad;
    at::Tensor value_grad;

    query_grad = OpPreparation::apply_tensor_without_format(format_query);
    key_grad = OpPreparation::apply_tensor_without_format(format_key);
    value_grad = OpPreparation::apply_tensor_without_format(format_value);

    char* input_layout_ptr = const_cast<char *>(input_layout.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendFlashAttentionGrad,
                                 format_query, format_key, format_value,
                                 format_softmax_log_max_sum, format_attention_score, format_attention_score_grad,
                                 format_atten_mask, format_alibi_mask, format_drop_mask,
                                 scale_value, head_num, input_layout_ptr, keep_prob, pre_tokens, next_tokens,
                                 query_grad, key_grad, value_grad);

    return std::make_tuple(query_grad, key_grad, value_grad);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_multi_head_attention_v2_grad(
    const at::Tensor &attention_score_grad,
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &softmax_log_max_sum,
    const at::Tensor &attention_score,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &alibi_mask,
    double scale,
    int64_t head_num,
    c10::string_view input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    bool gen_mask_parallel,
    bool sync)
{
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1,
        "The keep_prob value must be in range of [0, 1], but got ", keep_prob, OPS_ERROR(ErrCode::VALUE));

    std::string input_layout_str = std::string(input_layout);
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD",
        "The input_layout should be BSH/SBH/BNSD(case-insensitive), but got ", input_layout, OPS_ERROR(ErrCode::PARAM));

    int64_t length = (numels + 256 - 1) / 256 * 256 / 8;
    length += 32;
    at::Tensor drop_mask;
    if (get_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = gen_mask_dispatch(query, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
                                      gen_mask_parallel, sync);
    } else if (get_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, query.options().dtype(at::kByte));
    }

    auto results = npu_multi_head_attention_v2_backward(attention_score_grad, query, key, value,
                                                        softmax_log_max_sum, attention_score,
                                                        atten_mask, alibi_mask, drop_mask,
                                                        scale, head_num, input_layout_str, keep_prob,
                                                        pre_tokens, next_tokens, gen_mask_parallel, sync);

    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }
    return results;
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_multi_head_attention_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask_opt,
    const c10::optional<at::Tensor> &alibi_mask_opt,
    double scale,
    int64_t head_num,
    c10::string_view input_layout,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    bool gen_mask_parallel,
    bool sync)
{
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
    const at::Tensor &alibi_mask = alibi_mask_opt.value_or(at::Tensor());

    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1,
        "The keep_prob value must be in range of [0, 1], but got ", keep_prob, OPS_ERROR(ErrCode::VALUE));

    std::string input_layout_str = std::string(input_layout);
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH" || input_layout_str == "BNSD",
        "The input_layout should be BSH/SBH/BNSD(case-insensitive), but got ", input_layout, OPS_ERROR(ErrCode::PARAM));
    char* input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    int64_t seed;
    int64_t offset;
    int64_t numels;

    at::Tensor drop_mask = gen_mask(query, keep_prob, head_num, input_layout_str,
                                    gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor format_drop_mask = tensor_format_trans(drop_mask);
    at::Tensor format_query = tensor_format_trans(query);
    at::Tensor format_key = tensor_format_trans(key);
    at::Tensor format_value = tensor_format_trans(value);
    at::Tensor format_atten_mask = tensor_format_trans(atten_mask);
    at::Tensor format_alibi_mask = tensor_format_trans(alibi_mask);

    int64_t B = 0;
    int64_t S = 0;
    if (input_layout == "BSH") {
        B = query.size(0);
        S = query.size(1);
    } else if (input_layout == "SBH") {
        B = query.size(1);
        S = query.size(0);
    } else if (input_layout == "BNSD") {
        B = query.size(0);
        S = query.size(2);
    }
    at::Tensor softmax_log_max_sum = OpPreparation::apply_tensor_without_format({B, head_num, S},
                                                                                query.options().dtype(at::kFloat));
    at::Tensor attention_score = OpPreparation::apply_tensor_without_format(query);

    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendFlashAttention,
                                 format_query, format_key, format_value, format_atten_mask, format_alibi_mask,
                                 format_drop_mask, scale, head_num, input_layout_ptr, keep_prob, pre_tokens,
                                 next_tokens, softmax_log_max_sum, attention_score);

    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    return std::make_tuple(attention_score, softmax_log_max_sum, seed, offset, numels);
}
}
