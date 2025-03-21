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

namespace {
at::Tensor& rotated_iou_npu_nocheck(
    at::Tensor& iou,
    const at::Tensor& boxes,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold)
{
    string mode_str = (mode == 0) ? "iou" : "iof";

    at_npu::native::OpCommand cmd;
    cmd.Name("RotatedIou")
        .Input(boxes)
        .Input(query_boxes)
        .Output(iou)
        .Attr("trans", trans)
        .Attr("mode", mode_str)
        .Attr("is_cross", is_cross)
        .Attr("value", static_cast<float>(v_threshold))
        .Attr("value", static_cast<float>(e_threshold))
        .Run();
    return iou;
}
} // namespace

at::Tensor npu_rotated_iou(
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold)
{
    TORCH_CHECK(self.ndimension() == 3 && query_boxes.ndimension() == 3, OPS_ERROR(ErrCode::PARAM));

    auto origin_dtype = self.scalar_type();

    at::Tensor boxes_cp = self.permute({0, 2, 1});
    if (origin_dtype == at::kHalf) {
        boxes_cp = at_npu::native::custom_ops::npu_dtype_cast(boxes_cp, at::kFloat);
    }
    at::Tensor query_boxes_cp = query_boxes.permute({0, 2, 1});
    if (query_boxes_cp.scalar_type() == at::kHalf) {
        query_boxes_cp = at_npu::native::custom_ops::npu_dtype_cast(query_boxes_cp, at::kFloat);
    }

    int64_t B = boxes_cp.size(0);
    int64_t N = boxes_cp.size(-1);
    int64_t K = query_boxes_cp.size(-1);

    c10::SmallVector<int64_t, SIZE> output_size({B, N, K});
    at::Tensor iou = npu_preparation::apply_tensor(boxes_cp, output_size);

    rotated_iou_npu_nocheck(iou, boxes_cp, query_boxes_cp, trans, mode, is_cross, v_threshold, e_threshold);
    iou = at_npu::native::custom_ops::npu_dtype_cast(iou, origin_dtype);
    return iou;
}
} // namespace acl_op
