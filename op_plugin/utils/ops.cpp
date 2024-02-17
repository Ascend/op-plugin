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

#include "op_plugin/include/ops.h"
#include "op_plugin/AclOpsInterface.h"

namespace at_npu {
namespace native {
at::Tensor npu_dropout_gen_mask(const at::Tensor &self, at::IntArrayRef size, double p, int64_t seed, int64_t offset,
                                c10::optional<bool> parallel, c10::optional<bool> sync)
{
    return acl_op::_npu_dropout_gen_mask(self, size, p, seed, offset, parallel, sync);
}
}  // namespace native
}  // namespace at_npu
