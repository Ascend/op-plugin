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
at::Tensor& rotated_overlaps_npu_nocheck(
    at::Tensor& overlaps,
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("RotatedOverlaps")
        .Input(self)
        .Input(query_boxes)
        .Output(overlaps)
        .Attr("trans", trans)
        .Run();
    return overlaps;
}
} // namespace

at::Tensor npu_rotated_overlaps(
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans)
{
    TORCH_CHECK(self.ndimension() == 3 && query_boxes.ndimension() == 3,
                "boxes' dim should be equal to query_boxes' ndimension() ",
                "and equal to 3!" + OPS_ERROR(ErrCode::PARAM));
    auto origin_dtype = self.scalar_type();
    // the Op only support fp32 currently!
    at::Tensor self_cp = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat).permute({0, 2, 1});
    at::Tensor query_boxes_cp = at_npu::native::custom_ops::npu_dtype_cast(query_boxes, at::kFloat).permute({0, 2, 1});

    int64_t B = self_cp.size(0);
    int64_t N = self_cp.size(-1);
    int64_t K = query_boxes_cp.size(-1);

    c10::SmallVector<int64_t, SIZE> output_size({B, N, K});
    at::Tensor overlaps = npu_preparation::apply_tensor(self_cp, output_size);

    rotated_overlaps_npu_nocheck(overlaps, self_cp, query_boxes_cp, trans);
    overlaps = at_npu::native::custom_ops::npu_dtype_cast(overlaps, origin_dtype);
    return overlaps;
}
} // namespace acl_op
