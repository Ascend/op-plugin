// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

at::Tensor &reflection_pad1d_out(const at::Tensor &self, at::IntArrayRef padding, at::Tensor &result)
{
    TORCH_CHECK(padding.size() >= 2, "padding length shoud be at least 2" + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
    at::Tensor self_cp = self.unsqueeze(0);
    acl_op::reflection_pad2d_out(self_cp, paddings, result);
    result.squeeze_(0);
    return result;
}

at::Tensor reflection_pad1d(const at::Tensor &self, at::IntArrayRef padding)
{
    TORCH_CHECK(padding.size() >= 2, "padding length shoud be at least 2" + OPS_ERROR(ErrCode::PARAM));
    c10::SmallVector<int64_t, N> paddings = {padding[0], padding[1], 0, 0};
    at::Tensor self_cp = self.unsqueeze(0);
    at::Tensor result = acl_op::reflection_pad2d(self_cp, paddings);
    result.squeeze_(0);
    return result;
}

} // namespace acl_op
