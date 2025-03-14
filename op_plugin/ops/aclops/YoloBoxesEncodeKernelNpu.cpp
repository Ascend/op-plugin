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
const static int64_t MAX_ANCHOR_BOX_SIZE = 20480;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
inline void yolo_boxes_encode_check(
    const at::Tensor& anchor_boxes,
    const at::Tensor& gt_bboxes,
    const at::Tensor& stride)
{
    TORCH_CHECK(
        anchor_boxes.dim() == 2 && anchor_boxes.size(1) == 4,
        "Non-empty 2D anchor_boxes tensor expected but got a tensor with sizes ",
        anchor_boxes.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        anchor_boxes.size(0) <= MAX_ANCHOR_BOX_SIZE,
        "anchor_boxes only support max [20480] num, but got num ",
        anchor_boxes.size(0), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        gt_bboxes.dim() == 2 && gt_bboxes.size(1) == 4,
        "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
        gt_bboxes.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        stride.dim() == 1,
        "Non-empty 1D stride tensor expected but got a tensor with sizes ",
        stride.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        stride.size(0) == gt_bboxes.size(0),
        "stride's length should be equal gt_bboxes' num, but got stride length ",
        stride.size(0),
        "gt_bboxes num ",
        gt_bboxes.size(0), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        at::isIntegralType(stride.scalar_type(), true) && stride.scalar_type() != at::ScalarType::Long,
        "int32 strdie tensor expected but got a tensor with dtype: ",
        stride.scalar_type(), OPS_ERROR(ErrCode::TYPE));
}
} // namespace

at::Tensor npu_yolo_boxes_encode(
    const at::Tensor& self,
    const at::Tensor& gt_bboxes,
    const at::Tensor& stride,
    bool performance_mode)
{
    yolo_boxes_encode_check(self, gt_bboxes, stride);
    at::Tensor result = npu_preparation::apply_tensor(gt_bboxes);
    string impl_mode_str = performance_mode ? "high_performance" : "high_precision";
    at::Tensor stride_cp = at_npu::native::custom_ops::npu_dtype_cast(stride, at::ScalarType::Int);
    at_npu::native::OpCommand cmd;
    cmd.Name("YoloBoxesEncode")
        .Input(self)
        .Input(gt_bboxes)
        .Input(stride_cp)
        .Output(result)
        .Attr("performance_mode", impl_mode_str)
        .Run();
    return result;
}
} // namespace acl_op
