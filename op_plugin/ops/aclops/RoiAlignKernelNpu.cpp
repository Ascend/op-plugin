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
c10::SmallVector<int64_t, SIZE> roi_align_npu_output_size(const at::Tensor &self, const at::Tensor &rois,
                                                          int64_t pooled_height, int64_t pooled_width)
{
    TORCH_CHECK(rois.dim() >= 1, "The dim of input tensor [rois] is less than 1." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dim() >= 2, "The dim of input tensor [self] is less than 2." + OPS_ERROR(ErrCode::PARAM));
    return {rois.size(0), self.size(1), pooled_height, pooled_width}; // {N, C, H1, W1}
}

at::Tensor &roi_align_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &rois,
                                  double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num,
                                  int64_t roi_end_mode)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ROIAlign")
        .Input(self, "features")
        .Input(rois)
        .Output(result, "y")
        .Attr("spatial_scale", static_cast<float>(spatial_scale))
        .Attr("pooled_height", pooled_height)
        .Attr("pooled_width", pooled_width)
        .Attr("sample_num", sample_num)
        .Attr("roi_end_mode", roi_end_mode)
        .Run();

    return result;
}
} // namespace

at::Tensor npu_roi_align(const at::Tensor &self, const at::Tensor &rois, double spatial_scale, int64_t pooled_height,
                         int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode)
{
    at::Tensor self_cast = self;
    at::Tensor rois_cast = rois;
    if (self.scalar_type() == at::kHalf || rois.scalar_type() == at::kHalf) {
        self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::kFloat);
        rois_cast = at_npu::native::custom_ops::npu_dtype_cast(rois, at::kFloat);
    }

    auto output_size = roi_align_npu_output_size(self, rois, pooled_height, pooled_width);

    at::Tensor result = npu_preparation::apply_tensor_with_format(self, output_size, ACL_FORMAT_NC1HWC0);

    roi_align_npu_nocheck(result, self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);

    if (self.scalar_type() == at::kHalf || rois.scalar_type() == at::kHalf) {
        result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kHalf);
    }

    return result;
}

} // namespace acl_op
