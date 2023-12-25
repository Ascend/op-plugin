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


#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
const static int FLASH_THRESHOLD = 512;
const static int64_t DROPOUT_GEN_LEN = 128;
const static int64_t BYTE_BIT = 8;

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using namespace at_npu::native;

enum class DropOutStatus {
    DROPOUT_NORMAL = 0,
    DROPOUT_NONE,
    DROPOUT_ALL
};

DropOutStatus get_dropout_status(double keep_prob)
{
    if (keep_prob == 0) {
        return DropOutStatus::DROPOUT_ALL;
    }
    if (keep_prob == 1.) {
        return DropOutStatus::DROPOUT_NONE;
    }
    return DropOutStatus::DROPOUT_NORMAL;
}

at::Tensor format_trans(const at::Tensor &at_tensor)
{
    if (at_tensor.defined()) {
        TORCH_CHECK(torch_npu::utils::is_npu(at_tensor), "only npu tensor is supported");
        return custom_ops::npu_format_cast(at_tensor, ACL_FORMAT_ND);
    }
    return at_tensor;
}

at::Tensor dropout_gen_mask_impl(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
    const int64_t offset, const int64_t numels)
{
    int64_t length = (numels + DROPOUT_GEN_LEN - 1) / DROPOUT_GEN_LEN * DROPOUT_GEN_LEN / BYTE_BIT;
    c10::TensorOptions options = self.options();
    at::Tensor mask = OpPreparation::apply_tensor_without_format(at::IntArrayRef{length + 32},
        options.dtype(at::kByte));
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

at::Tensor dropout_gen_mask_dispatch(const at::Tensor &self, const at::Scalar &keep_prob, const at::Scalar &seed,
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
            mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
            if (sync) {
                NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(original_stream));
            }
        }
    } else {
        mask = dropout_gen_mask_impl(self, keep_prob, seed, offset, numels);
    }
    return mask;
}

at::Tensor dropout_gen_mask(const at::Tensor &self, double keep_prob, int64_t head_num, std::string input_layout,
    bool gen_mask_parallel, bool sync, int64_t &seed, int64_t &offset, int64_t &numels)
{
    at::Tensor drop_mask;
    if (input_layout == "BSH") {
        numels = self.size(0) * head_num * self.size(1) * self.size(1); // [B,N,S,S]
    } else if (input_layout == "SBH") {
        numels = self.size(1) * head_num * self.size(0) * self.size(0); // [B,N,S,S]
    }
    int64_t length = (numels + DROPOUT_GEN_LEN - 1) / DROPOUT_GEN_LEN * DROPOUT_GEN_LEN / BYTE_BIT;
    length += 32;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        const auto gen = at_npu::detail::getDefaultNPUGenerator();
        auto pair = at::check_generator<at_npu::NPUGeneratorImpl>(gen)->philox_engine_inputs(10);
        seed = pair.first;
        offset = pair.second;
        drop_mask = dropout_gen_mask_dispatch(self, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
            gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, self.options().dtype(at::kByte));
    }
    return drop_mask;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_backward_v3(
    // required input
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &qkv,
    const at::Tensor &dy,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    const at::Tensor &attention_in,
    // optional input
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef prefix,
    const c10::optional<at::Tensor> &drop_mask,

    // optional attrs
    double scale_qk,
    double scale_q,
    double scale_k,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode,
    int64_t head_num,
    const std::string input_layout,
    int64_t pse_type,
    int64_t head_size)
{
    const at::Tensor &pse_const = pse.value_or(at::Tensor());
    const at::Tensor &atten_mask_const = atten_mask.value_or(at::Tensor());
    const at::Tensor &drop_mask_const = drop_mask.value_or(at::Tensor());
    const at::Tensor &bias_const = bias.value_or(at::Tensor());
    auto prefix_n = prefix.value_or(at::IntArrayRef{});

    at::Tensor format_x = format_trans(x);
    at::Tensor format_wgt = format_trans(weight);
    at::Tensor format_qkv = format_trans(qkv);
    at::Tensor format_dy = format_trans(dy);

    at::Tensor format_pse = format_trans(pse_const);
    at::Tensor format_atten_mask = format_trans(atten_mask_const);
    at::Tensor format_drop_mask = format_trans(drop_mask_const);
    at::Tensor format_softmax_max = format_trans(softmax_max);
    at::Tensor format_softmax_sum = format_trans(softmax_sum);
    at::Tensor format_attention = format_trans(attention_in);
    at::Tensor format_bias = format_trans(bias_const);

    at::Tensor dx = OpPreparation::apply_tensor_without_format(format_x);
    at::Tensor dwgt = OpPreparation::apply_tensor_without_format(format_wgt);
    at::Tensor dpse;
    if (format_pse.defined()) {
        dpse = OpPreparation::apply_tensor_without_format(format_pse);
    } else {
        dpse = at::empty({0}, qkv.options());
    }
    at::Tensor dbias;
    if (format_bias.defined()) {
        dbias = OpPreparation::apply_tensor_without_format(format_bias);
    } else {
        dbias = at::empty({0}, qkv.options());
    }

    char* input_layout_ptr = const_cast<char *>(input_layout.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendAttentionGrad, format_x, format_wgt, format_qkv,
        format_dy, format_pse, format_atten_mask, prefix_n, format_drop_mask, format_softmax_max,
        format_softmax_sum, format_attention, format_bias, scale_qk, scale_q, scale_k,
        keep_prob, pre_tokens, next_tokens, sparse_mode, head_num, input_layout_ptr, pse_type,
        head_size, dx, dwgt, dpse, dbias);

    if (!format_pse.defined()) {
        at::Tensor dpse_required;
        dpse = dpse_required;
    }

    return std::make_tuple(dx, dwgt, dpse, dbias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad_v3(
    // required input
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &qkv,
    const at::Tensor &dy,
    const at::Tensor &softmax_max,
    const at::Tensor &softmax_sum,
    const at::Tensor &attention_in,

    // required attrs
    int64_t head_num,
    c10::string_view input_layout,
    int64_t head_size,

    // optional input
    const c10::optional<at::Tensor> &bias,
    const c10::optional<at::Tensor> &pse,
    const c10::optional<at::Tensor> &atten_mask,
    c10::OptionalIntArrayRef prefix,

    // optional attrs
    double scale_qk,
    double scale_q,
    double scale_k,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode,
    int64_t pse_type,
    int64_t seed,
    int64_t offset,
    int64_t numels,
    bool gen_mask_parallel,
    bool sync)
{
    TORCH_CHECK(x.dim() == 3, "The shapes of the input x should be 3-dimensional, but got ", x.dim(), "-dimensional");
    TORCH_CHECK(weight.dim() == 2, "The shapes of the input weight should be 2-dimensional, but got ", weight.dim(), "-dimensional");
    TORCH_CHECK(qkv.dim() == 3, "The shapes of the input qkv should be 3-dimensional, but got ", qkv.dim(), "-dimensional");
    TORCH_CHECK(dy.dim() == 3, "The shapes of the input dy should be 3-dimensional, but got ", dy.dim(), "-dimensional");
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
    std::string input_layout_str = std::string(input_layout);
    for (auto& c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
        "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout);

    int64_t length = (numels + DROPOUT_GEN_LEN - 1) / DROPOUT_GEN_LEN * DROPOUT_GEN_LEN / BYTE_BIT;
    length += 32;
    at::Tensor drop_mask;
    if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_NORMAL) {
        drop_mask = dropout_gen_mask_dispatch(qkv, at::Scalar(keep_prob), at::Scalar(seed), offset, numels,
            gen_mask_parallel, sync);
    } else if (get_dropout_status(keep_prob) == DropOutStatus::DROPOUT_ALL) {
        drop_mask = at::zeros(at::IntArrayRef{length}, qkv.options().dtype(at::kByte));
    }
    auto result = npu_fusion_attention_backward_v3(x, weight, qkv, dy, softmax_max,
        softmax_sum, attention_in, bias, pse, atten_mask, prefix, drop_mask, scale_qk,
        scale_q, scale_k, keep_prob, pre_tokens, next_tokens, sparse_mode, head_num, input_layout_str,
        pse_type, head_size);
    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }

    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_fusion_attention_v3(
    // required input
    const at::Tensor &x,
    const at::Tensor &weight,
    int64_t head_num,
    c10::string_view input_layout,
    int64_t head_size,
    // optional input
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<at::Tensor> &pse_opt,
    const c10::optional<at::Tensor> &atten_mask_opt,
    c10::OptionalIntArrayRef prefix,
    // optional attrs
    double scale_qk,
    double scale_q,
    double scale_k,
    double keep_prob,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t sparse_mode,
    int64_t pse_type,
    bool gen_mask_parallel,
    bool sync)
{
    const at::Tensor &bias = bias_opt.value_or(at::Tensor());
    const at::Tensor &pse = pse_opt.value_or(at::Tensor());
    const at::Tensor &atten_mask = atten_mask_opt.value_or(at::Tensor());
    auto prefix_n = prefix.value_or(at::IntArrayRef{});

    TORCH_CHECK(x.dim() == 3, "The shapes of the input x should be 3-dimensional, but got ", x.dim(), "-dimensional");
    TORCH_CHECK(weight.dim() == 2, "The shapes of the input weight should be 2-dimensional, but got ", weight.dim(), "-dimensional");
    TORCH_CHECK(keep_prob >= 0 && keep_prob <= 1, "The keep_prob value must be in range of [0, 1], but got ", keep_prob);
    std::string input_layout_str = std::string(input_layout);
    for (auto& c : input_layout_str) {
        c = toupper(c);
    }
    TORCH_CHECK(input_layout_str == "BSH" || input_layout_str == "SBH",
        "The input_layout should be BSH/SBH(case-insensitive), but got ", input_layout);

    int64_t B = 0;
    int64_t S0 = 0; // S for query
    int64_t S1 = 0; // S for key & value
    int64_t D = head_size;
    int64_t N1 = head_num;
    int64_t H1 = N1 * D;
    int64_t H2 = 0;
    int64_t N2 = 0;
    int64_t G = 1;

    if (input_layout_str == "BSH") {
        B = x.size(0);
        S0 = x.size(1);
        S1 = x.size(1);
        H2 = (weight.size(1) - H1) / 2;
        N2 = H2 / D;
    } else if (input_layout_str == "SBH") {
        B = x.size(1);
        S0 = x.size(0);
        S1 = x.size(0);
        H2 = (weight.size(1) - H1) / 2;
        N2 = H2 / D;
    }
    G = N1 / N2;

    double scale_qk_value = scale_qk;
    double scale_q_value = scale_q;
    double scale_k_value = scale_k;

    at::Tensor format_x = format_trans(x);
    at::Tensor attention_score = OpPreparation::apply_tensor_without_format({B, S0, H1}, x.options());
    if (input_layout_str == "SBH") {
        attention_score = OpPreparation::apply_tensor_without_format({S0, B, H1}, x.options());
    }
    at::Tensor format_weight = format_trans(weight);
    at::Tensor format_bias = format_trans(bias);

    at::Tensor format_pse = format_trans(pse);
    at::Tensor format_atten_mask = format_trans(atten_mask);

    int64_t seed;
    int64_t offset;
    int64_t numels;
    at::Tensor format_drop_mask = dropout_gen_mask(format_x, keep_prob, head_num, input_layout_str,
        gen_mask_parallel, sync, seed, offset, numels);

    at::Tensor softmax_max;
    at::Tensor softmax_sum;
    at::Tensor qkv;

    softmax_max = OpPreparation::apply_tensor_without_format({B, head_num, S0, 8},
        x.options().dtype(at::kFloat)); // [B, N, S0, 8]
    softmax_sum = OpPreparation::apply_tensor_without_format({B, head_num, S0, 8},
        x.options().dtype(at::kFloat)); // [B, N, S0, 8]

    qkv = OpPreparation::apply_tensor_without_format({B * N2 * (G + 2), S1, D},
        x.options());

    char* input_layout_ptr = const_cast<char *>(input_layout_str.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnAscendAttention, format_x, format_weight,
        format_bias, format_pse, format_drop_mask, prefix_n, format_atten_mask,
        scale_qk_value, scale_q_value, scale_k_value, keep_prob, pre_tokens, next_tokens, sparse_mode,
        head_num, input_layout_ptr, pse_type, head_size, softmax_max, softmax_sum, qkv, attention_score);
    if (!sync) {
        c10_npu::NPUEvent npu_event;
        npu_event.record(c10_npu::getCurrentNPUStream());
        npu_event.block(c10_npu::getCurrentSecondaryStream());
    }
    return std::make_tuple(attention_score, softmax_max, softmax_sum, qkv,
        seed, offset, numels);
}
} // namespace op_api