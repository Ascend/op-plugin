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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> npu_apply_rotary_pos_emb(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos,
    const at::Tensor &sin,
    c10::string_view layout)
{
    int64_t lay_out = 1;
    EXEC_NPU_CMD(aclnnApplyRotaryPosEmb, query, key, cos, sin, lay_out);
    return std::tie(query, key);
}

}
