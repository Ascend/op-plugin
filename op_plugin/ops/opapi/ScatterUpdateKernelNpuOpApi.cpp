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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
at::Tensor scatter_update(
    const at::Tensor &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    int64_t axis)
{
    // The attribute 'reduce' of Scatter only supports setting it to 'update'.
    at::Tensor result = self.clone();
    EXEC_NPU_CMD(aclnnInplaceScatterUpdate, result, indices, updates, axis);
    return result;
}

at::Tensor &scatter_update_(
    at::Tensor &self,
    const at::Tensor &indices,
    const at::Tensor &updates,
    int64_t axis)
{
    // The attribute 'reduce' of Scatter only supports setting it to 'update'.
    EXEC_NPU_CMD(aclnnInplaceScatterUpdate, self, indices, updates, axis);
    return self;
}
}  // namespace op_api
