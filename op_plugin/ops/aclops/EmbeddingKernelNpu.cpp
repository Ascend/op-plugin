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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
    return embedding_common_nocheck(weight, indices);
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor embedding_symint(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
    TORCH_CHECK(weight.device() == indices.device(),
        "Expected all tensors to be on the same device, but "
        "found at least two devices, ", weight.device(), " and ", indices.device(), "! "
        "(when checking argument for argument indices in method acl_op::embedding_symint)",
        OPS_ERROR(ErrCode::PARAM));
    return embedding_common_nocheck(weight, indices);
}
#endif

} // namespace acl_op
