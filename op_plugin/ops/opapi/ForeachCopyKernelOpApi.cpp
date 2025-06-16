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
const size_t SIZE_OF_NOT_INT = 4;
const size_t SIZE_OF_SHORT = 2;
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;


void exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    if (non_blocking) {
        EXEC_NPU_CMD(aclnnForeachCopy, src, dst);
    } else {
        OP_EXEC_LOG(aclnnForeachCopy, "EXEC_NPU_CMD_SYNC", src, dst);
        EXEC_NPU_CMD_SYNC(aclnnForeachCopy, src, dst);
    }
}

void split_and_exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    size_t tensor_count = src.size();
    size_t max_tensor_count = SINGLE_FOREACH_OP_TENSOR_COUNT;
    size_t loop_time = tensor_count / max_tensor_count;

    if (tensor_count <= max_tensor_count) {
        exec_npu_cmd_copy(dst, src, non_blocking);
        return;
    }
    for (size_t i = 0; i < loop_time; i++) {
        at::TensorList temp_src(src.data() + i * max_tensor_count, max_tensor_count);
        at::TensorList temp_dst(dst.data() + i * max_tensor_count, max_tensor_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }

    size_t remaining_count = tensor_count % max_tensor_count;
    if (remaining_count != 0) {
        at::TensorList temp_src(src.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_dst(dst.data() + loop_time * max_tensor_count, remaining_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }
}

bool check_tensor_dtype_support_base(const at::TensorList src)
{
    if ((sizeof(src[0]) == SIZE_OF_NOT_INT && src[0].scalar_type() != at::ScalarType::QInt32) ||
         src[0].scalar_type() == at::ScalarType::Int) {
        return true;
    }
    if (sizeof(src[0]) == SIZE_OF_SHORT || src[0].scalar_type() == at::ScalarType::Short) {
        return true;
    }
    if (src[0].scalar_type() == at::ScalarType::Char || src[0].scalar_type() == at::ScalarType::Byte ||
        src[0].scalar_type() == at::ScalarType::BFloat16 ||
        src[0].scalar_type() == at::ScalarType::Float || src[0].scalar_type() == at::ScalarType::Half) {
        return true;
    } else if (op_plugin::utils::is_gte_cann_version_810rc1() && (src[0].scalar_type() == at::ScalarType::Long ||
            src[0].scalar_type() == at::ScalarType::Double || src[0].scalar_type() == at::ScalarType::Bool)) {
        return true;
    }
    return false;
}

void _foreach_copy_(const at::TensorList self, const at::TensorList src, bool non_blocking)
{
    DO_COMPATIBILITY(aclnnForeachCopy, at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking));
    at::native::check_foreach_api_restrictions(self, src);
    static const bool is_support_nd_out = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||
                                          (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);
    if (!is_support_nd_out || !at::native::can_use_fast_route(self, src) || !check_tensor_dtype_support_base(src)) {
        return at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    }

    split_and_exec_npu_cmd_copy(self, src, non_blocking);
}

#endif
} // namespace at_npu
