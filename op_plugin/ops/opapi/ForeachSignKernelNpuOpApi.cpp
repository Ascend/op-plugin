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

#include <ATen/native/ForeachUtils.h>

#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include "op_plugin/utils/custom_functions/opapi/ForeachConstants.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;

void _split_and_exec_npu_cmd_sign(const at::TensorList tensors1, at::TensorList result_list, bool is_inplace)
{
    size_t tensor_count = tensors1.size();
    size_t max_tensor_count = is_inplace ? SINGLE_FOREACH_OP_TENSOR_COUNT : DOUBLE_FOREACH_OP_TENSOR_COUNT;

    size_t loop_time = tensor_count / max_tensor_count;
    if (tensor_count <= max_tensor_count) {
        EXEC_NPU_CMD(aclnnForeachSign, tensors1, result_list);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_tensors1(tensors1.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_result(result_list.data() + i * max_tensor_count, max_tensor_count);
        EXEC_NPU_CMD(aclnnForeachSign, temp_tensors1, temp_result);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count != 0) {
        at::TensorList temp_tensors1(tensors1.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_result(result_list.data() + loop_time * max_tensor_count, remaining_count);
        EXEC_NPU_CMD(aclnnForeachSign, temp_tensors1, temp_result);
    }
}

void _foreach_sign_(const at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachSign, at::native::foreach_tensor_sign_slow_(self));
    at::native::check_foreach_api_restrictions(self);

    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    bool is_support_type = op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32, op_plugin::utils::ForeachInputType::TYPE_TENSOR);
    if (op_plugin::utils::is_gte_cann_version_810rc1()) {
        is_support_type = op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::TO_INT, op_plugin::utils::ForeachInputType::TYPE_TENSOR);
    }
    if (!is_support_nd_out || !is_support_type || !at::native::can_use_fast_route(self)) {
        return at::native::foreach_tensor_sign_slow_(self);
    }
    
    _split_and_exec_npu_cmd_sign(self, self, true);
}

std::vector<at::Tensor> _foreach_sign(const at::TensorList self)
{
    DO_COMPATIBILITY(aclnnForeachSign, at::native::foreach_tensor_sign_slow(self));
    at::native::check_foreach_api_restrictions(self);

    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    bool is_support_type = op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::TO_INT32, op_plugin::utils::ForeachInputType::TYPE_TENSOR);
    if (op_plugin::utils::is_gte_cann_version_810rc1()) {
        is_support_type = op_plugin::utils::check_dtype_foreach(self[0].scalar_type(),
        op_plugin::utils::ForeachTensorDtypeSupport::TO_INT, op_plugin::utils::ForeachInputType::TYPE_TENSOR);
    }
    if (!is_support_nd_out || !is_support_type || !at::native::can_use_fast_route(self)) {
        return at::native::foreach_tensor_sign_slow(self);
    }
    auto scalar_type = self[0].scalar_type();

    // construct output tensorlist
    std::vector<at::Tensor> result;
    for (const at::Tensor &tensor : self) {
        auto output_size = op_infer::input_same_output_size(tensor);
        result.push_back(npu_preparation::apply_tensor_without_format(output_size,
                                                                      tensor.options().dtype(scalar_type)));
    }
    at::TensorList result_ = at::TensorList(result);

    _split_and_exec_npu_cmd_sign(self, result_, false);
    return result;
}
#endif

} // namespace at_npu
