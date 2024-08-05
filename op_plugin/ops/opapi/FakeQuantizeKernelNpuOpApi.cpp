// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/custom_functions/opapi/inner_compute_op_api.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
std::tuple<at::Tensor, at::Tensor> fake_quantize_per_channel_affine_cachemask(
    const at::Tensor& self,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max)
{
    at::Tensor out = npu_preparation::apply_tensor(self, self.sizes());
    at::Tensor mask = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(at::kBool), self);
    EXEC_NPU_CMD(aclnnFakeQuantPerChannelAffineCachemask, self, scale, zero_point, axis, quant_min, quant_max,
                 out, mask);
    return std::tie(out, mask);
}

std::tuple<at::Tensor, at::Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    const at::Tensor& self,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    const at::Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max)
{
    at::Tensor out = npu_preparation::apply_tensor(self, self.sizes());
    at::Tensor mask = npu_preparation::apply_tensor(self.sizes(), self.options().dtype(at::kBool), self);
    float fake_enabled = fake_quant_enabled.item().toFloat();
    EXEC_NPU_CMD(aclnnFakeQuantPerTensorAffineCachemask, self, scale, zero_point, fake_enabled, quant_min, quant_max,
                 out, mask);
    return std::tie(out, mask);
}

std::tuple<at::Tensor, at::Tensor> _fused_moving_avg_obs_fq_helper(
    const at::Tensor& self,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const double averaging_const,
    const int64_t quant_min,
    const int64_t quant_max,
    const int64_t ch_axis,
    bool per_row_fake_quant,
    bool symmetric_quant)
{
    return _fused_moving_avg_obs_fq_helper_common(self, observer_on, fake_quant_on, running_min, running_max,
        scale, zero_point, averaging_const, quant_min, quant_max, ch_axis, per_row_fake_quant, symmetric_quant);
}
#endif

} // namespace op_api
