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

namespace acl_op {
at::Tensor _embedding_bag_per_sample_weights_backward(const at::Tensor& grad, const at::Tensor& weight,
                                                      const at::Tensor& indices, const at::Tensor& offsets,
                                                      const at::Tensor& offset2bag, int64_t mode,
                                                      int64_t padding_idx)
{
    at::Tensor grad_cpu = grad.to("cpu");
    at::Tensor weight_cpu = weight.to("cpu");
    at::Tensor indices_cpu = indices.to("cpu");
    at::Tensor offsets_cpu = offsets.to("cpu");
    at::Tensor offset2bag_cpu = offset2bag.to("cpu");
    at::Tensor result = at::_embedding_bag_per_sample_weights_backward(grad_cpu, weight_cpu, indices_cpu,
                                                                       offsets_cpu, offset2bag_cpu, mode, padding_idx);
    result = at::native::sparse_to_dense(result);
    result = result.to(indices.device());
    return result;
}
} // namespace acl_op
