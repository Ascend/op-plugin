// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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
void _npu_paged_attention_mla(const at::Tensor &query, const at::Tensor &key_cache, int64_t num_kv_heads, int64_t num_heads, double scale_value, const at::Tensor &block_table, const at::Tensor &context_lens, int64_t mla_vheadsize, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;
    pagedparam.headNum = num_heads;
    pagedparam.qkScale = scale_value;
    pagedparam.kvHeadNum = num_kv_heads;
    auto mlavHeadSize = static_cast<uint32_t>(mla_vheadsize);
    pagedparam.mlaVHeadSize = mlavHeadSize;

    pagedparam.maskType = PagedAttentionParam::UNDEFINED;
    pagedparam.batchRunStatusEnable = false;
    pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    pagedparam.outDataType = ACL_DT_UNDEFINED;
    pagedparam.hasQuantOffset = false;
    pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
    pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
    pagedparam.inputLayout = atb::infer::TYPE_BSND;

    ParamSetter paramsetter;
    paramsetter.Input(query, true)
        .Input(key_cache)
        .Input(block_table, true)
        .Input(context_lens, true)
        .Output(out);
    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_attention_mla(Tensor query, Tensor key_cache, int num_kv_heads, int num_heads, float scale_value, Tensor block_table, Tensor context_lens, int mla_vheadsize, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_attention_mla", TORCH_FN(atb::_npu_paged_attention_mla));
}
}
}