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
at::Tensor& roi_align_backward_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& rois,
    at::IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num,
    c10::optional<int64_t> roi_end_mode)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ROIAlignGrad")
        .Input(self, "ydiff")
        .Input(rois)
        .Output(result, "xdiff")
        .Attr("xdiff_shape", xdiff_shape)
        .Attr("spatial_scale", static_cast<float>(spatial_scale))
        .Attr("pooled_height", pooled_height)
        .Attr("pooled_width", pooled_width)
        .Attr("sample_num", sample_num);
    if (roi_end_mode.has_value()) {
        cmd.Attr("roi_end_mode", roi_end_mode.value());
    }
    cmd.Run();

    return result;
}
} // namespace

at::Tensor npu_roi_alignbk(
    const at::Tensor& self,
    const at::Tensor& rois,
    at::IntArrayRef xdiff_shape,
    int64_t pooled_width,
    int64_t pooled_height,
    double spatial_scale,
    int64_t sample_num,
    c10::optional<int64_t> roi_end_mode)
{
    at::Tensor result =
        npu_preparation::apply_tensor_with_format(self, xdiff_shape, ACL_FORMAT_NC1HWC0);

    // Check the self empty
    for (int i = 0; i < self.dim(); i++) {
        if (self.size(i) == 0) {
            acl_op::fill_(result, 0);
            return result;
        }
    }

    roi_align_backward_npu_nocheck(
        result,
        self,
        rois,
        xdiff_shape,
        pooled_width,
        pooled_height,
        spatial_scale,
        sample_num,
        roi_end_mode);

    return result;
}

} // namespace acl_op
