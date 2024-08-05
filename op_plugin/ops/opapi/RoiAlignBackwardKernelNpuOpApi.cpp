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

at::Tensor npu_roi_align(const at::Tensor& self, const at::Tensor& rois, double spatial_scale, int64_t pooled_height,
                         int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode)
{
    DO_COMPATIBILITY(aclnnROIAlign, acl_op::npu_roi_align(self, rois, spatial_scale, pooled_height, pooled_width,
                                                          sample_num, roi_end_mode));
    TORCH_CHECK(rois.dim() >= 1, "The dim of input tensor [rois] is less than 1.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dim() >= 2, "The dim of input tensor [self] is less than 2.", OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, SIZE> output_size = {rois.size(0), self.size(1), pooled_height, pooled_width};
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    float spatial_scale_value = static_cast<float>(spatial_scale);
    EXEC_NPU_CMD(aclnnROIAlign, self, rois, spatial_scale_value, pooled_height, pooled_width, sample_num, roi_end_mode,
                 result);
    return result;
}
}
