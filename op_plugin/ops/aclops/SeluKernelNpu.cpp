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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& selu_out_npu_nocheck(at::Tensor& result, const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Selu")
        .Input(self)
        .Output(result)
        .Run();

    return result;
}

at::Tensor selu_out_nocheck(const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    selu_out_npu_nocheck(result, self);
    return result;
}

at::Tensor& selu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& result)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SeluGrad")
        .Input(grad_output)
        .Input(result)
        .Output(grad_input)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor selu_backward(const at::Tensor& grad_output, const at::Tensor& result)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output);
    selu_backward_npu_nocheck(grad_input, grad_output, result);
    return grad_input;
}

at::Tensor selu(const at::Tensor& self)
{
    return selu_out_nocheck(self);
}

at::Tensor& selu_(at::Tensor& self)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        at::Tensor result = selu_out_nocheck(contiguous_self);
        npu_utils::format_fresh_view(self, result);
    } else {
        auto result = selu_out_nocheck(self);
        self.copy_(result);
    }
    return self;
}
} // namespace acl_op
