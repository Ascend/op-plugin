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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& hardtanh_backward_out_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("HardtanhGrad")
        .Input(self)
        .Input(grad_output)
        .Output(grad_input)
        .Attr("max_val", max_val)
        .Attr("min_val", min_val)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor& hardtanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {grad_output, self},
        grad_input,
        self);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_grad_input = npu_utils::format_contiguous(grad_input);
        hardtanh_backward_out_npu_nocheck(contiguous_grad_input, grad_output, self, min_val, max_val);
        npu_utils::format_fresh_view(grad_input, contiguous_grad_input);
    } else {
        hardtanh_backward_out_npu_nocheck(grad_input, grad_output, self, min_val, max_val);
    }

    return grad_input;
  }

at::Tensor hardtanh_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val)
{
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input);
    return grad_input;
}
} // namespace acl_op
