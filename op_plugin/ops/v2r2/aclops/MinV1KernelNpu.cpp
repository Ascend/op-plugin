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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
at::Tensor npu_min_backward_symint(const at::Tensor &grad, int64_t dim, const at::Tensor &indices,
                                   c10::SymIntArrayRef sizes_symint, bool keepdim)
{
    at::IntArrayRef sizes = c10::asIntArrayRefUnchecked(sizes_symint);
    at::Tensor new_grad = grad;
    at::Tensor new_indices = indices;
    if (keepdim && sizes.size() > 0) {
        new_grad = grad.squeeze(dim);
        new_indices = indices.squeeze(dim);
    }
    auto grad_input = acl_op::npu_scatter(at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
    return grad_input;
}
} // namespace acl_op
