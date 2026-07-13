// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"
#include <acl/acl.h>

using namespace std;

namespace atb {
using PagedAttentionParam = atb::infer::PagedAttentionParam;

// 将 PagedAttentionParam::MaskType 枚举值转为可读字符串，用于错误提示
static std::string MaskTypeToString(int64_t mtype)
{
    switch (static_cast<PagedAttentionParam::MaskType>(mtype)) {
        case PagedAttentionParam::UNDEFINED:              return "UNDEFINED(0)";
        case PagedAttentionParam::MASK_TYPE_NORM:         return "MASK_TYPE_NORM(1)";
        case PagedAttentionParam::MASK_TYPE_ALIBI:        return "MASK_TYPE_ALIBI(2)";
        case PagedAttentionParam::MASK_TYPE_SPEC:         return "MASK_TYPE_SPEC(3)";
        case PagedAttentionParam::MASK_TYPE_MASK_FREE:    return "MASK_TYPE_MASK_FREE(4)";
        case PagedAttentionParam::MASK_TYPE_NORM_COMPRESS:return "MASK_TYPE_NORM_COMPRESS(5)";
        default:                                          return "UNKNOWN(" + std::to_string(mtype) + ")";
    }
}

void _npu_paged_attention_splitfuse_v2(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &value_cache, const at::Tensor &block_table, const at::Tensor &context_lens, const at::Tensor &mask, const at::Tensor &seq_len, int64_t num_kv_heads, int64_t num_heads, double scale_value, const c10::optional<int64_t> mask_type, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    int64_t mtype = mask_type.has_value() ? mask_type.value() : PagedAttentionParam::MASK_TYPE_NORM_COMPRESS;
    TORCH_CHECK(mtype == PagedAttentionParam::MASK_TYPE_NORM_COMPRESS,
                "mask_type only supports MASK_TYPE_NORM_COMPRESS(5), but got: ", MaskTypeToString(mtype));
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;
    pagedparam.headNum = num_heads;
    pagedparam.qkScale = scale_value;
    pagedparam.kvHeadNum = num_kv_heads;
    pagedparam.maskType = static_cast<PagedAttentionParam::MaskType>(mtype);
    pagedparam.batchRunStatusEnable = false;
    pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    pagedparam.outDataType = ACL_DT_UNDEFINED;
    pagedparam.hasQuantOffset = false;
    pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    pagedparam.calcType = PagedAttentionParam::CALC_TYPE_SPEC;
    pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
    pagedparam.inputLayout = atb::infer::TYPE_BSND;
    pagedparam.mlaVHeadSize = 0;

    ParamSetter paramsetter;
    paramsetter.Input(query, true)
        .Input(key_cache)
        .Input(value_cache)
        .Input(block_table, true)
        .Input(context_lens, true)
        .Input(mask)
        .Input(seq_len, true)
        .Output(out);
    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_attention_splitfuse_v2(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor context_lens, Tensor mask, Tensor seq_len, int num_kv_heads, int num_heads, float scale_value, *, int ? mask_type = 5, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_attention_splitfuse_v2", TORCH_FN(atb::_npu_paged_attention_splitfuse_v2));
}
}

}
