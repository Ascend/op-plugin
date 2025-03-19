// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse)
{
    DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse));
    // calculate the output size
    auto output_size = op_infer::array_to_small_vector(indices.sizes());
    output_size.emplace_back(weight.size(weight.dim() - 1));
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
    // calculate the output resugt of the NPU
    EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse)
{
    DO_COMPATIBILITY(aclnnEmbedding, acl_op::embedding_symint(weight, indices, padding_idx,
                                                              scale_grad_by_freq, sparse));
    TORCH_CHECK(weight.device() == indices.device(),
        "Expected all tensors to be on the same device, but "
        "found at least two devices, ", weight.device(), " and ", indices.device(), "! "
        "(when checking argument for argument indices in method opapi::embedding_symint)",
        OPS_ERROR(ErrCode::PARAM));
    // calculate the output size
    auto output_size = op_infer::array_to_small_vector(indices.sizes());
    output_size.emplace_back(weight.size(weight.dim() - 1));
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, weight.options());
    // calculate the output resugt of the NPU
    EXEC_NPU_CMD(aclnnEmbedding, weight, indices, result);
    return result;
}
#endif

} // namespace op_api
