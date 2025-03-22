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
constexpr int DEFAULT_SCALES = -1;

at::Tensor &upsample_linear1d_out(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
                                  c10::optional<double> scales, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnUpsampleLinear1d,
                     acl_op::upsample_linear1d_out(self, output_size, align_corners, scales, out));

    auto outsize = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, out, self, outsize);
    double scales_h_attr = scales.value_or(DEFAULT_SCALES);

    EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, output_size, align_corners, scales_h_attr, out);
    return out;
}

at::Tensor upsample_linear1d(const at::Tensor &self, at::IntArrayRef output_size, bool align_corners,
                             c10::optional<double> scales)
{
    DO_COMPATIBILITY(aclnnUpsampleLinear1d, acl_op::upsample_linear1d(self, output_size, align_corners, scales));

    auto outsize = op_infer::upsample_linear1d_npu_output_size(self, output_size);
    double scales_h_attr = scales.value_or(DEFAULT_SCALES);

    at::Tensor result = npu_preparation::apply_tensor_without_format(outsize, self.options());

    EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, output_size, align_corners, scales_h_attr, result);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_linear1d(const at::Tensor &self, c10::optional<at::IntArrayRef> output_size, bool align_corners,
                             c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleLinear1d, acl_op::upsample_linear1d(self, output_size, align_corners, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(self.sizes(), output_size, scale_factors);
    auto scales = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto outsize = at::IntArrayRef(osize);
    auto out_size = op_infer::upsample_linear1d_npu_output_size(self, outsize);
    constexpr int DEFAULT_SCALES = -1;
    double scales_h_attr = scales.value_or(DEFAULT_SCALES);
    at::Tensor result = npu_preparation::apply_tensor_without_format(out_size, self.options());

    EXEC_NPU_CMD(aclnnUpsampleLinear1d, self, outsize, align_corners, scales_h_attr, result);
    return result;
}
#endif

} // namespace op_api
