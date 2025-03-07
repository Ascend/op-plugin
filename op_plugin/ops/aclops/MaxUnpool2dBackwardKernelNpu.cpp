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

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& max_unpool2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {self, grad_output},
        grad_input,
        self);
    TORCH_CHECK(
        output_size.size() == 2,
        "There should be exactly two elements (height, width) in outputSize", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (self.ndimension() == 3 || self.ndimension() == 4),
        "Input to max_unpooling2d should be a 3d or 4d Tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        self.sizes() == indices.sizes(),
        "Shape of indices should match shape of input", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.numel() > 0, "Input must be non-empty", OPS_ERROR(ErrCode::PARAM));

    auto oheight = output_size[0];
    auto owidth = output_size[1];
    int64_t n = 1;
    int64_t c = self.size(0);
    int64_t h = self.size(1);
    int64_t w = self.size(2);
    int64_t self_dim = self.ndimension();
    if (self_dim == 4) {
        n = self.size(0);
        c = self.size(1);
        h = self.size(2);
        w = self.size(3);
    }

    auto grad_output_contiguous = grad_output.contiguous();
    auto indices_contiguous = indices.contiguous();
    grad_output_contiguous = grad_output_contiguous.reshape({n, c, oheight * owidth});
    indices_contiguous = indices_contiguous.reshape({n, c, h * w});
    grad_input.resize_as_(self);
    grad_input.zero_();
    grad_input = grad_input.reshape({n, c, h * w});
    const int dim = 2;

    grad_input = acl_op::gather_out(grad_output_contiguous, dim, indices_contiguous, false, grad_input);
    if (self_dim == 3) {
        grad_input = grad_input.reshape({c, h, w});
    } else {
        grad_input = grad_input.reshape({n, c, h, w});
    }
    return grad_input;
}

at::Tensor max_unpool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size)
{
    auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    max_unpool2d_backward_out(grad_output, self, indices, output_size, grad_input);
    return grad_input;
}
#endif

} // namespace acl_op
