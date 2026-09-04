// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_moe_token_permute_grad_v2_symint(
    const at::Tensor &grad_permuted_tokens,
    const at::Tensor &sorted_indices,
    c10::SymInt tokens_size_0,
    at::ScalarType tokens_dtype,
    c10::SymInt num_topK,
    bool padded_mode)
{
    int64_t tokens_size_0_value = tokens_size_0.expect_int();
    int64_t num_topK_value = num_topK.expect_int();
    auto output_size_0 = op_infer::npu_moe_token_permute_grad_v2_out_size(
        tokens_size_0_value, grad_permuted_tokens, sorted_indices);
    at::Tensor grad_tokens = npu_preparation::apply_tensor_without_format(
        output_size_0, grad_permuted_tokens.options().dtype(tokens_dtype));
    EXEC_NPU_CMD(aclnnMoeTokenPermuteGrad, grad_permuted_tokens, sorted_indices, num_topK_value, padded_mode, grad_tokens);
    return grad_tokens;
}

}  // namespace op_api
