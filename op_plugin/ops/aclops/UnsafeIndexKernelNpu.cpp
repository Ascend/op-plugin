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

#include "op_plugin/utils/OpAdapter.h"
#if VERSION_BETWEEN(V2R1, V2R1)
#include "op_plugin/AclOpsInterface.h"
#include <ATen/ops/_unsafe_index_native.h>

namespace acl_op {

at::Tensor _unsafe_index(const at::Tensor &self, const torch::List<c10::optional<at::Tensor>> &indices)
{
    return at::native::_unsafe_index(self, indices);
}
} // namespace acl_op
#endif
