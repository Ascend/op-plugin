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

#include <torch/csrc/autograd/custom_function.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using torch::autograd::AutogradContext;

namespace {
at::Tensor& fast_gelu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FastGeluGrad")
        .Input(grad)
        .Input(self)
        .Output(grad_input)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor npu_fast_gelu(const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    at_npu::native::OpCommand cmd;
    cmd.Name("FastGelu")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}

at::Tensor npu_fast_gelu_backward(
    const at::Tensor& grad,
    const at::Tensor& self)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    fast_gelu_backward_npu_nocheck(grad_input, grad, self);
    return grad_input;
}

at::Tensor fast_gelu(const at::Tensor& self)
{
    return acl_op::npu_fast_gelu(self);
}
} // namespace acl_op
