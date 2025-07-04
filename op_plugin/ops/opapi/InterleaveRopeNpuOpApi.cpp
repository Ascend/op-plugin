// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

    at::Tensor npu_interleave_rope(
        const at::Tensor &x,
        const at::Tensor &cos,
        const at::Tensor &sin)
    {
        at::Tensor out = npu_preparation::apply_tensor_without_format(x.sizes(), x.options());
        EXEC_NPU_CMD(aclnnInterleaveRope, x, cos, sin, out);
        return out;
    }
} // namespace op_api
