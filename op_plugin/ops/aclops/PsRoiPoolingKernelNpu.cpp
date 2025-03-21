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
using npu_op_command = at_npu::native::OpCommand;

namespace {
at::Tensor& ps_roi_pooling_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim)
{
    npu_op_command cmd;
    cmd.Name("PSROIPoolingV2")
        .Input(self, "x")
        .Input(rois)
        .Output(result, "y")
        .Attr("spatial_scale", static_cast<float>(spatial_scale))
        .Attr("output_dim", output_dim)
        .Attr("group_size", group_size)
        .Run();

    return result;
}

at::Tensor &ps_roi_pooling_backward_npu_nocheck(at::Tensor &input_grad, const at::Tensor &output_grad,
                                                const at::Tensor &rois, double spatial_scale, int64_t group_size,
                                                int64_t output_dim, at::IntArrayRef input_size)
{
    npu_op_command cmd;
    cmd.Name("PSROIPoolingGradV2D")
        .Input(output_grad, "x")
        .Input(rois)
        .Output(input_grad, "y")
        .Attr("spatial_scale", static_cast<float>(spatial_scale))
        .Attr("group_size", group_size)
        .Attr("output_dim", output_dim)
        .Attr("input_size", input_size)
        .Run();

    return input_grad;
}
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor npu_ps_roi_pooling_backward(const at::Tensor &output_grad, const at::Tensor &rois, double spatial_scale,
                                       int64_t group_size, int64_t output_dim, at::IntArrayRef input_size)
{
    TORCH_CHECK(rois.dim() > 0, "rois must has dim > 0", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(input_size.size() >= 2, "input_size must has dim >= 2", OPS_ERROR(ErrCode::PARAM));
    auto output_size = {rois.size(0), group_size * group_size * output_dim, input_size[0], input_size[1]};

    at::Tensor input_grad = npu_preparation::apply_tensor(output_grad, output_size);
    ps_roi_pooling_backward_npu_nocheck(input_grad, output_grad, rois, spatial_scale, group_size, output_dim,
                                        input_size);

    return input_grad;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor npu_ps_roi_pooling_backward_symint(const at::Tensor &output_grad, const at::Tensor &rois,
                                              double spatial_scale, int64_t group_size, int64_t output_dim,
                                              c10::SymIntArrayRef input_size)
{
    at::IntArrayRef input_size_int = c10::asIntArrayRefUnchecked(input_size);
    TORCH_CHECK(input_size_int.size() >= 2,
        "The length of param 'input_size' must be greater than or equal to 2." + OPS_ERROR(ErrCode::PARAM));
    auto output_size = {rois.size(0), group_size * group_size * output_dim, input_size_int[0], input_size_int[1]};

    at::Tensor input_grad = npu_preparation::apply_tensor(output_grad, output_size);
    ps_roi_pooling_backward_npu_nocheck(input_grad, output_grad, rois, spatial_scale, group_size, output_dim,
        input_size_int);

    return input_grad;
}
#endif

at::Tensor npu_ps_roi_pooling(
    const at::Tensor& self,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t group_size,
    int64_t output_dim)
{
    TORCH_CHECK(rois.dim() >= 3,
        "rois only supports at least 3D tensors, rois got: ", rois.dim(), "D"
        + OPS_ERROR(ErrCode::PARAM));

    auto output_size = {rois.size(0) * rois.size(2), output_dim, group_size, group_size};
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    ps_roi_pooling_npu_nocheck(result, self, rois, spatial_scale, group_size, output_dim);
    return result;
}

} // namespace acl_op
