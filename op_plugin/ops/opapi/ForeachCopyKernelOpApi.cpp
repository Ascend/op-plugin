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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"
#include <ATen/native/ForeachUtils.h>
#include "op_plugin/utils/custom_functions/opapi/ForeachConstants.h"

namespace op_api {
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;
using npu_calcu_util = at_npu::native::CalcuOpUtil;


void exec_npu_cmd_copy(const at::TensorList dst, at::TensorList src, bool non_blocking)
{
    if (non_blocking) {
        EXEC_NPU_CMD(aclnnForeachCopy, src, dst);
    } else {
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
    if (remaining_count) {
        at::TensorList temp_src(src.data() + loop_time * max_tensor_count, remaining_count);
        at::TensorList temp_dst(dst.data() + loop_time * max_tensor_count, remaining_count);
        exec_npu_cmd_copy(temp_dst, temp_src, non_blocking);
    }
}

void _foreach_copy_(const at::TensorList self, const at::TensorList src, bool non_blocking)
{
    DO_COMPATIBILITY(aclnnForeachCopy, at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking));
    at::native::check_foreach_api_restrictions(self, src);

    static const bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&
                                          c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1;
    if (!is_support_nd_out || !at::native::can_use_fast_route(self, src)) {
        return at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    }

    split_and_exec_npu_cmd_copy(self, src, non_blocking);
}

#endif
} // namespace at_npu
