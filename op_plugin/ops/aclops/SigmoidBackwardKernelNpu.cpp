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
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& sigmoid_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& output)
{
    auto unified_result = npu_preparation::binary_op_check(result, output, grad_output, true);
    at_npu::native::OpCommand cmd;
    cmd.Name("SigmoidGrad")
        .Expect(unified_result)
        .Input(output)
        .Input(grad_output)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor& sigmoid_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut({grad_output, output}, grad_input, grad_output);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        sigmoid_backward_out_npu_nocheck(contiguous_result, grad_output, output);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        sigmoid_backward_out_npu_nocheck(grad_input, grad_output, output);
    }
    return grad_input;
}

at::Tensor sigmoid_backward(const at::Tensor& grad_output, const at::Tensor& output)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(grad_output);
    sigmoid_backward_out_npu_nocheck(grad_input, grad_output, output);

    return grad_input;
}

} // namespace acl_op
