// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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
using npu_utils = at_npu::native::NpuUtils;

at::Tensor &__ilshift__(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnLeftShift, acl_op::__ilshift__(self, other));
    EXEC_NPU_CMD(aclnnLeftShift, self, other, self);
    return self;
}

at::Tensor &__ilshift__(at::Tensor &self, const at::Scalar &other)
{
    DO_COMPATIBILITY(aclnnLeftShifts, acl_op::__ilshift__(self, other));
    EXEC_NPU_CMD(aclnnLeftShifts, self, other, self);
    return self;
}
} // namespace op_api