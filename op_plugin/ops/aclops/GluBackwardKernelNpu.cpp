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
// See the License for the specific language govern_ing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
void glu_grad_npu_check(const at::Tensor& self, int64_t dim)
{
    TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional Tensors"
        + OPS_ERROR(ErrCode::NOT_SUPPORT));
    auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
    const int64_t n_in = self.size(wrap_dim);
    TORCH_CHECK(n_in % 2 == 0, "Halving dimension must be even, but dimension ",
        wrap_dim, " is size ", n_in, OPS_ERROR(ErrCode::NOT_SUPPORT));
}

at::Tensor& glu_grad_npu_out_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("GLUGrad")
        .Input(grad_output)
        .Input(self)
        .Output(grad_input)
        .Attr("dim", dim)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor& glu_backward_out(const at::Tensor& grad_output, const at::Tensor& self, int64_t dim, at::Tensor& grad_input)
{
    glu_grad_npu_check(self, dim);
    auto output_size = op_infer::input_same_output_size(self);
    npu_preparation::CheckOut(
        {grad_output, self},
        grad_input,
        grad_output,
        output_size);

    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        glu_grad_npu_out_nocheck(contiguous_result, grad_output, self, dim);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        glu_grad_npu_out_nocheck(grad_input, grad_output, self, dim);
    }
    return grad_input;
}

at::Tensor glu_backward(const at::Tensor& grad_output, const at::Tensor& self, int64_t dim)
{
    glu_grad_npu_check(self, dim);
    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    glu_grad_npu_out_nocheck(grad_input, grad_output, self, dim);
    return grad_input;
}
} // namespace acl_op
