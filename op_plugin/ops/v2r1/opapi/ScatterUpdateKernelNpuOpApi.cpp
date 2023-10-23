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

at::Tensor scatter_update(
    const at::Tensor& data,
    const at::Tensor& indices,
    const at::Tensor& updates,
    int64_t axis)
{
    DO_COMPATIBILITY(aclnnInplaceScatterUpdate, acl_op::scatter_update(data, indices, updates, axis));
    TORCH_NPU_WARN_ONCE(
        "Warning: kernel [scatter_update] is a out-of-place op, but it is supported by another in-place op cann.Scatter."
        "This current usage may cause the input to be changed unexpectedly, "
        "and the caller needs to pay attention to this feature.");
    EXEC_NPU_CMD(aclnnInplaceScatterUpdate, data, indices, updates, axis);
    return data;
}

} // namespace op_api
