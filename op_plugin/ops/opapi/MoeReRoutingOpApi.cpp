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
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_moe_re_routing(
        const at::Tensor &tokens,
        const at::Tensor &expert_token_num_per_rank,
        const c10::optional <at::Tensor> &per_token_scales_opt,
        int64_t expert_token_num_type,
        int64_t idx_type,
        c10::optional<int64_t> tokens_dtype)
    {
        TORCH_CHECK(tokens.dim() > 1, "tokens dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(expert_token_num_per_rank.dim() > 1, "expert_token_num_per_rank dim should larger than 1", OPS_ERROR(ErrCode::PARAM));

        if (tokens_dtype.has_value()) {
            TORCH_CHECK(tokens_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8) ||
                        tokens_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                        tokens_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2),
                "The optional parameter tokens_dtype only supports hifloat8, float4_e2m1fn_x2, float4_e1m2fn_x2 or None, but now is ",
                c10_npu::CustomDataTypeToString(tokens_dtype.value()),
                "." + OPS_ERROR(ErrCode::VALUE));
        }

        at::SmallVector <int64_t, op_infer::SIZE> permute_tokens_size;
        at::SmallVector <int64_t, op_infer::SIZE> permute_per_token_scales_size;
        at::SmallVector <int64_t, op_infer::SIZE> permute_token_idx_size;
        at::SmallVector <int64_t, op_infer::SIZE> expert_token_num_size;
        for (int i = 0; i < tokens.dim(); i++) {
            permute_tokens_size.push_back(tokens.size(i));
        }
        const at::Tensor &per_token_scales = c10::value_or_else(per_token_scales_opt, [] { return at::Tensor(); });
        auto per_token_scales_dtype = per_token_scales.dtype();
        if (per_token_scales.defined()) {
            for (int i = 0; i < per_token_scales.dim(); i++) {
                permute_per_token_scales_size.push_back(per_token_scales.size(i));
            }
        } else {
            permute_per_token_scales_size.push_back(tokens.size(0));
        }
        permute_token_idx_size.push_back(tokens.size(0));
        expert_token_num_size.push_back(expert_token_num_per_rank.size(1));

        at::TensorOptions permute_tokens_options = tokens.options();
        if (tokens_dtype.has_value()) {
            aclDataType acl_dtype = c10_npu::GetAclDataType(tokens_dtype.value());
            if (acl_dtype == aclDataType::ACL_FLOAT4_E2M1 || acl_dtype == aclDataType::ACL_FLOAT4_E1M2) {
                permute_tokens_options = permute_tokens_options.dtype(at::kByte);
            } else {
                permute_tokens_options = permute_tokens_options.dtype(c10_npu::GetATenDType(tokens_dtype.value()));
            }
        }
        at::Tensor permute_tokens = npu_preparation::apply_tensor_without_format(permute_tokens_size, permute_tokens_options);
        at::Tensor permute_per_token_scales =
            per_token_scales.defined()
                ? npu_preparation::apply_tensor_without_format(permute_per_token_scales_size, per_token_scales.options())
                : npu_preparation::apply_tensor_without_format(permute_per_token_scales_size, c10::dtype(c10::ScalarType::Float));
        at::Tensor permute_token_idx = npu_preparation::apply_tensor_without_format(permute_token_idx_size, c10::dtype(c10::ScalarType::Int));
        at::Tensor expert_token_num = npu_preparation::apply_tensor_without_format(expert_token_num_size, expert_token_num_per_rank.options());

        TensorWrapper tokens_wrapper = make_wrapper(tokens, tokens_dtype);
        TensorWrapper permute_tokens_wrapper = make_wrapper(permute_tokens, tokens_dtype);

        if (per_token_scales_dtype == at::kByte) {
            TensorWrapper per_token_scales_wrapper = {per_token_scales, aclDataType::ACL_FLOAT8_E8M0};
            TensorWrapper permute_per_token_scales_wrapper = {permute_per_token_scales, aclDataType::ACL_FLOAT8_E8M0};
            EXEC_NPU_CMD(aclnnMoeReRouting, tokens_wrapper, expert_token_num_per_rank, per_token_scales_wrapper,
                         expert_token_num_type, idx_type, permute_tokens_wrapper, permute_per_token_scales_wrapper,
                         permute_token_idx, expert_token_num);
        } else {
            EXEC_NPU_CMD(aclnnMoeReRouting, tokens_wrapper, expert_token_num_per_rank, per_token_scales,
                         expert_token_num_type, idx_type, permute_tokens_wrapper, permute_per_token_scales,
                         permute_token_idx, expert_token_num);
        }
        return std::tie(permute_tokens, permute_per_token_scales, permute_token_idx, expert_token_num);
    }
}