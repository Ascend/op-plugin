// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include <ATen/native/ForeachUtils.h>
#include "op_plugin/utils/custom_functions/opapi/ForeachConstants.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

bool check_zero_is_supported_data_type(at::ScalarType scalar_type)
{
    return (scalar_type == at::ScalarType::Half || scalar_type == at::ScalarType::Float ||
            scalar_type == at::ScalarType::Int || scalar_type == at::ScalarType::Short ||
            scalar_type == at::ScalarType::BFloat16);
}

void _split_and_exec_npu_cmd_zero(at::TensorList tensors1)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = SINGLE_FOREACH_OP_TENSOR_COUNT;

    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachZeroInplace, tensors1);
        return;
    }
    size_t loop_time = tensor_count / max_tensor_count;
    
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachZeroInplace, temp_tensors1);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachZeroInplace, temp_tensors1);
    }
}

void _foreach_zero_(const at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachZeroInplace, at::native::foreach_tensor_zero_slow_(self));
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out) {
        return at::native::foreach_tensor_zero_slow_(self);
    }

    at::native::check_foreach_api_restrictions(self);
    if (!at::native::can_use_fast_route(self) || at::native::has_integral_tensor(self, true)) {
        return at::native::foreach_tensor_zero_slow_(self);
    }

    if (self.empty()) {
        return;
    }

    auto scalar_type = self[0].scalar_type();
    TORCH_CHECK(op_api::check_zero_is_supported_data_type(scalar_type),
        "input must be half, float, int or short or bfloat16", OPS_ERROR(ErrCode::TYPE));

    _split_and_exec_npu_cmd_zero(self);
}
} // namespace at_npu
