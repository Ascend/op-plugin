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
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor, at::Tensor> prelu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    at::Tensor grad_weight = npu_preparation::apply_tensor(weight);
    prelu_backward_commom_nocheck(grad_input, grad_weight, grad_output, self, weight);
    return std::tie<at::Tensor, at::Tensor>(grad_input, grad_weight);
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
std::tuple<at::Tensor, at::Tensor> _prelu_kernel_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight)
{
    c10::SmallVector<int64_t, N> weight_shape = op_infer::array_to_small_vector(weight.sizes());
    at::Tensor reshape_weight = weight.reshape({-1});

    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    at::Tensor grad_weight = npu_preparation::apply_tensor(reshape_weight);

    prelu_backward_commom_nocheck(grad_input, grad_weight, grad_output, self, reshape_weight);
    grad_weight = grad_weight.reshape(weight_shape);

    return std::tie<at::Tensor, at::Tensor>(grad_input, grad_weight);
}
#endif
} // namespace acl_op
