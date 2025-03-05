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

namespace acl_op {
at::Tensor gelu_common_nocheck(const at::Tensor& self)
{
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor(self);
    at_npu::native::OpCommand cmd;
    cmd.Name("Gelu")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}
} // namespace acl_op
