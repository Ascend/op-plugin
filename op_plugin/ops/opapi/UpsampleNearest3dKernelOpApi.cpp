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

at::Tensor& upsample_nearest3d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest3d,
        acl_op::upsample_nearest3d_out(self, output_size, scales_d, scales_h, scales_w, out));
    auto output_osize = op_infer::upsample_nearest3d_npu_output_size(self, output_size);
    npu_preparation::check_tensor({self}, out, self, output_osize);
    double scales_d_attr = scales_d.value_or(0);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleNearest3d, self, output_size, scales_d_attr, scales_h_attr, scales_w_attr, out);
    return out;
}

at::Tensor upsample_nearest3d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest3d,
        acl_op::upsample_nearest3d(self, output_size, scales_d, scales_h, scales_w));
    auto output_osize = op_infer::upsample_nearest3d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_osize);
    double scales_d_attr = scales_d.value_or(0);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);
    EXEC_NPU_CMD(aclnnUpsampleNearest3d, self, output_size, scales_d_attr, scales_h_attr, scales_w_attr, result);
    return result;
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor upsample_nearest3d(
    const at::Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors)
{
    DO_COMPATIBILITY(aclnnUpsampleNearest3d,
        acl_op::upsample_nearest3d(input, output_size, scale_factors));
    auto osize = op_infer::upsample_infershape_with_scale(input.sizes(), output_size, scale_factors);

    auto scales_d = op_plugin::utils::get_scale_value(scale_factors, 0);
    auto scales_h = op_plugin::utils::get_scale_value(scale_factors, 1);
    auto scales_w = op_plugin::utils::get_scale_value(scale_factors, 2);
    double scales_d_attr = scales_d.value_or(0);
    double scales_h_attr = scales_h.value_or(0);
    double scales_w_attr = scales_w.value_or(0);

    auto output_size_vec = op_infer::upsample_nearest3d_npu_output_size(input, osize);
    at::Tensor result = npu_preparation::apply_tensor_without_format(input, output_size_vec);
    auto output_osize = at::IntArrayRef(osize);

    EXEC_NPU_CMD(aclnnUpsampleNearest3d, input, output_osize, scales_d_attr, scales_h_attr, scales_w_attr, result);
    return result;
}
#endif
}
