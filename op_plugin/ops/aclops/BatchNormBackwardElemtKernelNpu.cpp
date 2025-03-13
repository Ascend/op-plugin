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
void batch_norm_backward_elemt_npu_expand_tensor(
    at::Tensor& expand_tensor,
    int64_t dim_c,
    int64_t input_ndim,
    at::IntArrayRef input_shape)
{
    if (input_ndim > 2) {
        expand_tensor = acl_op::npu_broadcast(expand_tensor, {1, dim_c}).t();
        for (int64_t i = 0; i < input_ndim - 3; i++) {
        expand_tensor = expand_tensor.unsqueeze(1);
        }
    }
    expand_tensor = acl_op::npu_broadcast(expand_tensor, input_shape);
}
} // namespace

at::Tensor batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count)
{
    const at::Tensor& weight_value = c10::value_or_else(weight, [] {return at::Tensor();});
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 1, "input.dim() <= 1" + OPS_ERROR(ErrCode::PARAM));

    auto divisor = count.sum();
    TORCH_CHECK(divisor.numel() > 0, "The input tensor [count] is an empty tensor."
        + OPS_ERROR(ErrCode::PARAM));
    auto mean_dy = sum_dy.div(divisor);
    auto mean_dy_xmu = sum_dy_xmu.div(divisor);

    int64_t dim_c = input.size(1);
    at::IntArrayRef input_shape = input.sizes();
    at::Tensor mean_expanded(mean);

    batch_norm_backward_elemt_npu_expand_tensor(mean_expanded, dim_c, input_ndim, input_shape);
    at::Tensor invstd_expanded(invstd);

    batch_norm_backward_elemt_npu_expand_tensor(invstd_expanded, dim_c, input_ndim, input_shape);
    at::Tensor weight_expanded(weight_value);

    batch_norm_backward_elemt_npu_expand_tensor(weight_expanded, dim_c, input_ndim, input_shape);
    at::Tensor mean_dy_expanded(mean_dy);

    batch_norm_backward_elemt_npu_expand_tensor(mean_dy_expanded, dim_c, input_ndim, input_shape);
    at::Tensor mean_dy_xmu_expanded(mean_dy_xmu);

    batch_norm_backward_elemt_npu_expand_tensor(mean_dy_xmu_expanded, dim_c, input_ndim, input_shape);
    at::Tensor grad_input = npu_preparation::apply_tensor(input);

    at_npu::native::OpCommand cmd;
    cmd.Name("SyncBatchNormBackwardElemt")
        .Input(grad_out)
        .Input(input)
        .Input(mean_expanded)
        .Input(invstd_expanded)
        .Input(weight_expanded)
        .Input(mean_dy_expanded)
        .Input(mean_dy_xmu_expanded)
        .Output(grad_input)
        .Run();

    return grad_input;
}
} // namespace acl_op
