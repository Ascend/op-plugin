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
#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& max_unpool3d_backward_out_npu_nocheck(at::Tensor& grad_input, const at::Tensor& grad_output,
                                                  const at::Tensor& indices)
{
    int64_t N = 1;
    int64_t C = indices.size(0);
    if (grad_output.dim() == 5) {
        N = indices.size(0);
        C = indices.size(1);
    }
    at::Tensor reshape_grad_output = grad_output.reshape({N, C, -1});
    at::Tensor reshape_indices = indices.reshape({N, C, -1});
    grad_input = grad_input.reshape({N, C, -1});

    int64_t dim = 2;
    at_npu::native::OpCommand cmd;
    cmd.Name("GatherElements")
        .Input(reshape_grad_output)
        .Input(reshape_indices)
        .Output(grad_input)
        .Attr("dim", dim)
        .Run();
    grad_input = grad_input.reshape(indices.sizes());
    return grad_input;
}
#endif
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& max_unpool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input)
{
    TORCH_CHECK(output_size.size() == 3, "There should be exactly 3 elements (depth, height, width) in output_size",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
                "Input to max_unpooling2d should be a 4d or 5d Tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.sizes() == indices.sizes(), "Shape of indices should match shape of input", OPS_ERROR(ErrCode::PARAM));
    npu_preparation::CheckOut({grad_output, self, indices}, grad_input, self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        max_unpool3d_backward_out_npu_nocheck(contiguous_result, grad_output, indices);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);
    }
    return grad_input;
}

at::Tensor max_unpool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding)
{
    TORCH_CHECK(output_size.size() == 3, "There should be exactly 3 elements (depth, height, width) in output_size",
                OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
                "Input to max_unpooling2d should be a 4d or 5d Tensor", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.sizes() == indices.sizes(), "Shape of indices should match shape of input", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.numel() > 0, "Input must be non-empty", OPS_ERROR(ErrCode::PARAM));

    at::Tensor grad_input = npu_preparation::apply_tensor(self);
    max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);
    return grad_input;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& max_unpool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {grad_output, self, indices},
        grad_input,
        self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        max_unpool3d_backward_out_npu_nocheck(contiguous_result, grad_output, indices);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);
    }
    return grad_input;
}

at::Tensor max_unpool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding)
{
    TORCH_CHECK(
        output_size.size() == 3,
        "There should be exactly 3 elements (depth, height, width) in output_size");
    TORCH_CHECK(
        (self.ndimension() == 4 || self.ndimension() == 5),
        "Input to max_unpooling2d should be a 4d or 5d Tensor");
    TORCH_CHECK(
        self.sizes() == indices.sizes(),
        "Shape of indices should match shape of input");
    TORCH_CHECK(self.numel() > 0, "Input must be non-empty");

    at::Tensor grad_input = npu_preparation::apply_tensor(self);

    max_unpool3d_backward_out_npu_nocheck(grad_input, grad_output, indices);

    return grad_input;
}
#endif
} // namespace acl_op
