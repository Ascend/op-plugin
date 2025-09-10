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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
at::Tensor leaky_relu_backward_out_nocheck(
    at::Tensor result,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negval)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("LeakyReluGrad")
        .Input(grad_output)
        .Input(self)
        .Output(result)
        .Attr("negative_slope", negval)
        .Run();
    return result;
}
} // namespace

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
} // namespace acl_op
