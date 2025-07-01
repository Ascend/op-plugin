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
constexpr int64_t LAYOUT_BSND_BSH = 1;
constexpr int64_t LAYOUT_SBND = 2;
constexpr int64_t LAYOUT_BNSD = 3;
std::tuple<at::Tensor, at::Tensor> _apply_rotary_pos_emb_v1(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos,
    const at::Tensor &sin,
    int64_t lay_out)
{
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnApplyRotaryPosEmb, query, key, cos, sin, lay_out);
    return std::tie(query, key);
}

std::tuple<at::Tensor, at::Tensor> npu_apply_rotary_pos_emb(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos,
    const at::Tensor &sin,
    c10::string_view layout,
    c10::string_view rotary_mode)
{
    std::string layout_str = std::string(layout);
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "SBND" || layout_str == "BSH",
        "The layout should be BSND/BSH/BNSD/SBND, but got ", layout_str, OPS_ERROR(ErrCode::PARAM));
    std::string rotary_mode_str = std::string(rotary_mode);
    TORCH_CHECK(rotary_mode_str == "half" || rotary_mode_str == "quarter" || rotary_mode_str == "interleave",
        "The layout should be half/quarter/interleave, but got ", rotary_mode_str, OPS_ERROR(ErrCode::PARAM));
    int64_t lay_out = LAYOUT_BSND_BSH;
    if (layout_str == "BNSD") {
        lay_out = LAYOUT_BNSD;
    } else if (layout_str == "SBND") {
        lay_out = LAYOUT_SBND;
    }
    DO_COMPATIBILITY(aclnnApplyRotaryPosEmbV2, _apply_rotary_pos_emb_v1(query, key, cos, sin, lay_out));

    char* rotary_mode_ptr = const_cast<char *>(rotary_mode_str.c_str());
    EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnnApplyRotaryPosEmbV2, query, key, cos, sin, lay_out, rotary_mode_ptr);
    return std::tie(query, key);
}
}
