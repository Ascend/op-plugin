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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& upsample_nearest1d_old_out(const at::Tensor& self,
                                       at::IntArrayRef output_size,
                                       c10::optional<double> scales,
                                       at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest1d, acl_op::upsample_nearest1d_out(self, output_size, scales, result));
    c10::SmallVector<int64_t, SIZE> out_size = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, result, self, out_size);
    EXEC_NPU_CMD(aclnnUpsampleNearest1d, self, output_size, result);
    return result;
}

at::Tensor upsample_nearest1d_old(const at::Tensor& self,
                                  at::IntArrayRef output_size,
                                  c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest1d, acl_op::upsample_nearest1d(self, output_size, scales));
    c10::SmallVector<int64_t, SIZE> out_size = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, out_size);
    EXEC_NPU_CMD(aclnnUpsampleNearest1d, self, output_size, result);
    return result;
}
at::Tensor& upsample_nearest1d_out(const at::Tensor& self,
                                   at::IntArrayRef output_size,
                                   c10::optional<double> scales,
                                   at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest1dV2, op_api::upsample_nearest1d_old_out(self, output_size, scales, result));
    c10::SmallVector<int64_t, SIZE> out_size = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, result, self, out_size);
    float scale_l = static_cast<float>(scales.value_or(-1.0));
    EXEC_NPU_CMD(aclnnUpsampleNearest1dV2, self, output_size, scale_l, result);
    return result;
}

at::Tensor upsample_nearest1d(const at::Tensor& self,
                              at::IntArrayRef output_size,
                              c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest1dV2, op_api::upsample_nearest1d_old(self, output_size, scales));
    float scale_l = static_cast<float>(scales.value_or(-1.0));
    c10::SmallVector<int64_t, SIZE> out_size = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, out_size);
    EXEC_NPU_CMD(aclnnUpsampleNearest1dV2, self, output_size, scale_l, result);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest1d(const at::Tensor& input,
                              c10::optional<at::IntArrayRef> output_size,
                              c10::optional<at::ArrayRef<double>> scale_factors)
{
    // 兼容性处理,没有v2回调原先版本
    DO_COMPATIBILITY(aclnnUpsampleNearest1dV2, op_api::upsample_nearest1d_old(input, output_size, scale_factors));
    auto compute_size = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);
    return op_api::upsample_nearest1d(input, compute_size, scales_w);
}

at::Tensor upsample_nearest1d_old(const at::Tensor& input,
                                  c10::optional<at::IntArrayRef> output_size,
                                  c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest1d, acl_op::upsample_nearest1d(input, output_size, scale_factors));
    auto compute_size = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 0);
    return op_api::upsample_nearest1d_old(input, compute_size, scales_w);
}
#endif
}
