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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor>;

at::Tensor& multinomial_top_k_top_p_sample_op_api(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator)
{
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto is_capture = c10_npu::currentStreamCaptureStatusMayInitCtx();
    if (is_capture == c10_npu::CaptureStatus::None) {
        auto pair = gen->philox_engine_inputs(10);
        const uint64_t seed = pair.first;
        const uint64_t offset = pair.second;
        EXEC_NPU_CMD(aclnnMultinomial, self, num_samples, replacement, seed, offset, result);
    } else {
#if VERSION_BETWEEN(V2R5, VERSION_NEWEST)
        auto gen_state_ = gen->philox_npu_state(10);
        const at::Tensor* seed_ptr = gen_state_.seed_.ptr;
        const at::Tensor* offset_ptr = gen_state_.offset_.ptr;
        const uint64_t offset_intragraph = gen_state_.offset_intragraph_;
        EXEC_NPU_CMD(aclnnMultinomialTensor, self, num_samples, replacement, *seed_ptr, *offset_ptr, offset_intragraph, result);
#endif
    }
    return result;
}

at::Tensor multinomial_top_k_top_p_sample(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator)
{
    DO_COMPATIBILITY(aclnnMultinomial, acl_op::multinomial(self, num_samples, replacement, generator));
    auto dim = self.dim();
    auto shape = op_infer::array_to_small_vector(self.sizes());
    shape[dim-1] = num_samples;
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(shape, self.options().dtype(at::kLong));
    multinomial_top_k_top_p_sample_op_api(result, self, num_samples, replacement, generator);
    return result;
}

tensor_list npu_top_k_top_p_sample(const at::Tensor &logits, const at::Tensor &top_k, const at::Tensor &top_p, const c10::optional<at::Tensor> &q_option,
                                   const c10::optional<at::Tensor> &min_ps_option, c10::optional<double> eps_option, c10::optional<bool> is_need_logits_option,
                                   c10::optional<int64_t> top_k_guess_option, c10::optional<int64_t> ks_max_potion, c10::optional<bool> input_is_logits_option,
                                   c10::optional<c10::string_view> post_sample_option, c10::optional<at::Generator> generator)
{
    const at::Tensor &q = c10::value_or_else(q_option, [] { return at::Tensor(); });
    const at::Tensor &min_ps = c10::value_or_else(min_ps_option, [] { return at::Tensor(); });
    double eps = c10::value_or_else(eps_option, [] {return 1e-8;});
    bool is_need_logits = c10::value_or_else(is_need_logits_option, [] {return false; });
    int64_t top_k_guess = c10::value_or_else(top_k_guess_option, [] {return 32;});
    int64_t ks_max = c10::value_or_else(ks_max_potion, [] {return 1024;});
    bool input_is_logits = c10::value_or_else(input_is_logits_option, [] {return true; });
    c10::string_view post_sample = post_sample_option.value_or("qSample");

    /* output shape construct */
    auto logits_size = logits.sizes();
    auto batch = logits_size[0];
    auto voc_size = logits_size[1];

    bool is_need_sample_result = false;

    at::Tensor logits_select_idx = npu_preparation::apply_tensor_without_format({batch, }, logits.options().dtype(at::kLong));
    at::Tensor logits_top_kp_select = npu_preparation::apply_tensor_without_format({batch, voc_size}, logits.options().dtype(at::kFloat));

    at::Tensor logits_idx = npu_preparation::apply_tensor_without_format({batch, voc_size}, logits.options().dtype(at::kLong));
    at::Tensor logits_sort_masked = npu_preparation::apply_tensor_without_format({batch, voc_size}, logits.options().dtype(at::kFloat));

    std::string post_sample_str = std::string(post_sample);
    if (post_sample_str == "multiNomial") {
        is_need_sample_result = true;
        EXEC_NPU_CMD(aclnnTopKTopPSampleV2, logits, top_k, top_p, q, min_ps, eps, is_need_logits, top_k_guess, ks_max, input_is_logits, is_need_sample_result, logits_select_idx, logits_top_kp_select, logits_idx, logits_sort_masked);
        at::Tensor multinomial_result = multinomial_top_k_top_p_sample(logits_sort_masked, 1, true, generator);
        for (uint32_t i = 0; i < batch; ++i) {
            int64_t real_index = logits_idx[i][multinomial_result[i].item<int64_t>()].item<int64_t>();
            logits_select_idx[i] = real_index;
        }
    } else {
        EXEC_NPU_CMD(aclnnTopKTopPSampleV2, logits, top_k, top_p, q, min_ps, eps, is_need_logits, top_k_guess, ks_max, input_is_logits, is_need_sample_result, logits_select_idx, logits_top_kp_select, logits_idx, logits_sort_masked);
    }

    return std::tie(logits_select_idx, logits_top_kp_select);
}
}
