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
inline void anchor_response_flags_check(
    const at::Tensor& self,
    at::IntArrayRef featmap_size)
{
    TORCH_CHECK(
        featmap_size.size() == 2,
        "expected feat_map_size equals to 2, but got size ",
        featmap_size.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        self.dim() == 2 && self.size(1) == 4,
        "Non-empty 2D gt_bboxes tensor expected but got a tensor with sizes ",
        self.sizes(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        self.scalar_type() == at::kHalf || self.scalar_type() == at::kFloat,
        "float16 or float32 tensor expected but got a tensor with dtype: ",
        self.scalar_type(), OPS_ERROR(ErrCode::TYPE));
}
} // namespace

at::Tensor npu_anchor_response_flags(
    const at::Tensor& self,
    at::IntArrayRef featmap_size,
    at::IntArrayRef stride,
    int64_t num_base_anchors)
{
    anchor_response_flags_check(self, featmap_size);
    auto output_size = op_infer::infersize_npu_anchor_response_flags(featmap_size, num_base_anchors);
    auto options = self.options().dtype(at::kByte);
    at::Tensor result = npu_preparation::apply_tensor(output_size, options, self);
    at::Tensor self_cp = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);

    at_npu::native::OpCommand cmd;
    cmd.Name("AnchorResponseFlags")
        .Input(self_cp)
        .Output(result)
        .Attr("featmap_size", featmap_size)
        .Attr("strides", stride)
        .Attr("num_base_anchors", num_base_anchors)
        .Run();
    return result;
}
} // namespace acl_op
