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

at::Tensor leaky_relu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negval,
    bool is_result)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    leaky_relu_backward_out_nocheck(result, grad_output, self, negval);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& leaky_relu_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negval,
    bool is_result,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {self, grad_output},
        grad_input,
        grad_input,
        self.sizes());
    leaky_relu_backward_out_nocheck(grad_input, grad_output, self, negval);
    return grad_input;
}
#endif
} // namespace acl_op
