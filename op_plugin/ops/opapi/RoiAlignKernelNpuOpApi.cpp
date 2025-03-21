// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_roi_alignbk(const at::Tensor& self, const at::Tensor& rois, at::IntArrayRef xdiff_shape,
                           int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num,
                           c10::optional<int64_t> roi_end_mode)
{
    DO_COMPATIBILITY(
        aclnnROIAlignBackward,
        acl_op::npu_roi_alignbk(
            self,
            rois,
            xdiff_shape,
            pooled_width,
            pooled_height,
            spatial_scale,
            sample_num,
            roi_end_mode));
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, xdiff_shape);
    int64_t roi_end_mode_value = 1; // roi_end_mode default value
    if (roi_end_mode.has_value()) {
        roi_end_mode_value = roi_end_mode.value();
    }
    float spatial_scale_value = static_cast<float>(spatial_scale);
    EXEC_NPU_CMD(aclnnROIAlignBackward, self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale_value,
                 sample_num, roi_end_mode_value, result);
    return result;
}
}
