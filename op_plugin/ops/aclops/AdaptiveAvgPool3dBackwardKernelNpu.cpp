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
int64_t adaptive_avg_pool3d_backward_safe_size(const at::Tensor& self)
{
    c10::SmallVector<int64_t, N> dims = {-3, -2, -1};
    int64_t size = 1;
    if (self.sizes().empty()) {
        return size;
    }
    for (int64_t ndim : dims) {
        ndim = op_plugin::utils::make_warp_dim(ndim, self.sizes().size());
        size *= self.sizes()[ndim];
    }
    return size;
}

at::Tensor& adaptive_avg_pool3d_backward_out_nocheck(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& self)
{
    TORCH_CHECK(grad_output.size(grad_output.dim() - 3) == 1 && grad_output.size(grad_output.dim() - 2) == 1 &&
        grad_output.size(grad_output.dim() - 1) == 1,
        "adaptive_avg_pool3d_backward only support D=1 && H=1 && W=1 current!" + OPS_ERROR(ErrCode::PARAM));
    acl_op::fill_(result, 1.0 / adaptive_avg_pool3d_backward_safe_size(self));
    acl_op::mul_(result, grad_output);

    return result;
}
} // namespace

at::Tensor& adaptive_avg_pool3d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& grad_input)
{
    npu_preparation::CheckOut(
        {grad_output, self},
        grad_input,
        self);
    if (!npu_utils::check_match(&grad_input)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(grad_input);
        adaptive_avg_pool3d_backward_out_nocheck(contiguous_result, grad_output, self);
        npu_utils::format_fresh_view(grad_input, contiguous_result);
    } else {
        adaptive_avg_pool3d_backward_out_nocheck(grad_input, grad_output, self);
    }
    return grad_input;
}

at::Tensor _adaptive_avg_pool3d_backward(const at::Tensor& grad_output, const at::Tensor& self)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    adaptive_avg_pool3d_backward_out_nocheck(result, grad_output, self);
    return result;
}
} // namespace acl_op
