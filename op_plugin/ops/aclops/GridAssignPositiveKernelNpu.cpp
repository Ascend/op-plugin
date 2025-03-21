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

namespace {
static inline void grid_assign_positive_check(const at::Tensor& argmax_overlaps,
    const at::Tensor& gt_argmax_overlaps)
{
    TORCH_CHECK(
        at::isIntegralType(argmax_overlaps.scalar_type(), true) &&
        argmax_overlaps.scalar_type() != at::ScalarType::Long,
        "int32 argmax_overlaps tensor expected but got a tensor with dtype: ", argmax_overlaps.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(
        at::isIntegralType(gt_argmax_overlaps.scalar_type(), true) &&
            gt_argmax_overlaps.scalar_type() != at::ScalarType::Long,
        "int32 gt_argmax_overlaps tensor expected but got a tensor with dtype: ", gt_argmax_overlaps.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
}
}  // namespace

at::Tensor npu_grid_assign_positive(
    const at::Tensor& self,
    const at::Tensor& overlaps,
    const at::Tensor& box_responsible_flags,
    const at::Tensor& max_overlaps,
    const at::Tensor& argmax_overlaps,
    const at::Tensor& gt_max_overlaps,
    const at::Tensor& gt_argmax_overlaps,
    int64_t num_gts,
    double pos_iou_thr,
    double min_pos_iou,
    bool gt_max_assign_all)
{
    grid_assign_positive_check(argmax_overlaps, gt_argmax_overlaps);
    at::Tensor result = npu_preparation::apply_tensor(self);
    auto option = self.options().dtype(at::kInt);

    at::Scalar s(num_gts);
    auto num = at::empty({}, option);
    at::Tensor num_of_gts = acl_op::fill_(num, s);
    at::Tensor argmax_overLaps = at_npu::native::custom_ops::npu_dtype_cast(argmax_overlaps, at::ScalarType::Int);
    at::Tensor gt_argmax_overLaps = at_npu::native::custom_ops::npu_dtype_cast(gt_argmax_overlaps, at::ScalarType::Int);

    at_npu::native::OpCommand cmd;
    cmd.Name("GridAssignPositive")
        .Input(self)
        .Input(overlaps)
        .Input(box_responsible_flags)
        .Input(max_overlaps)
        .Input(argmax_overLaps)
        .Input(gt_max_overlaps)
        .Input(gt_argmax_overLaps)
        .Input(num_of_gts)
        .Output(result)
        .Attr("pos_iou_thr", (float)pos_iou_thr)
        .Attr("min_pos_iou", (float)min_pos_iou)
        .Attr("gt_max_assign_all", gt_max_assign_all)
        .Run();

    return result;
}
}  // op_plugin
