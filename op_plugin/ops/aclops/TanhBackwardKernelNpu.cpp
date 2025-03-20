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
at::Tensor& tanh_backward_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("TanhGrad")
        .Input(self)
        .Input(grad_output)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& tanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& result)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(grad_output, self);
    npu_preparation::CheckOut({grad_output, self}, result, self, output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        tanh_backward_out_npu_nocheck(contiguous_result, grad_output, self);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        tanh_backward_out_npu_nocheck(result, grad_output, self);
    }

    return result;
}

at::Tensor tanh_backward(const at::Tensor& grad_output, const at::Tensor& self)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(grad_output, self);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    tanh_backward_out_npu_nocheck(result, grad_output, self);
    return result;
}
} // namespace acl_op
