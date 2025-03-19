// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& hardshrink_backward_out(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Scalar& lambd,
    at::Tensor& grad_input)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("HardShrinkGrad")
       .Input(grad_out)
       .Input(self)
       .Attr("lambd", lambd)
       .Output(grad_input)
       .Run();
    return grad_input;
}

at::Tensor hardshrink_backward(
    const at::Tensor& grad_out,
    const at::Tensor& self,
    const at::Scalar& lambd)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    hardshrink_backward_out(grad_out, self, lambd, grad_input);
    return grad_input;
}
} // namespace acl_op
