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
#include <acl/acl.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/custom_functions/atb/AtbCommon.h"

using namespace std;

namespace atb {

using PagedAttentionParam = atb::infer::PagedAttentionParam;
at::Tensor& _npu_paged_attention_v2(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &block_table,
    c10::SymIntArrayRef context_lens,
    const c10::optional<at::Tensor> &value_cache,
    const c10::optional<at::Tensor> &mask,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    int64_t mask_type,
    at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;
    pagedparam.headNum = num_heads;
    pagedparam.qkScale = scale_value;
    pagedparam.kvHeadNum = num_kv_heads;
    auto masktype = static_cast<PagedAttentionParam::MaskType>(mask_type);
    pagedparam.maskType = masktype;
    pagedparam.batchRunStatusEnable = false;
    pagedparam.quantType = PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    pagedparam.outDataType = ACL_DT_UNDEFINED;
    pagedparam.hasQuantOffset = false;
    pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
    pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
    pagedparam.inputLayout = atb::infer::TYPE_BSND;
    pagedparam.mlaVHeadSize = 0;

    ParamSetter paramsetter;
    at::Tensor context_lens_tensor = at::tensor(c10::asIntArrayRefUnchecked(context_lens), at::kInt);
    if (pagedparam.maskType == PagedAttentionParam::UNDEFINED) {
        paramsetter.Input(query)
            .Input(key_cache)
            .Input(value_cache)
            .Input(block_table)
            .Input(context_lens_tensor)
            .Output(out);
    } else if (pagedparam.maskType == PagedAttentionParam::MASK_TYPE_ALIBI) {
        paramsetter.Input(query)
            .Input(key_cache)
            .Input(value_cache)
            .Input(block_table)
            .Input(context_lens_tensor)
            .Input(mask)
            .Output(out);
    }

    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return out;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_attention_v2.out(Tensor query, Tensor key_cache, Tensor block_table, SymInt[] context_lens, *, Tensor? value_cache=None, Tensor? mask=None, int num_kv_heads=0, int num_heads=0, float scale_value=1.0, int mask_type=0, Tensor(a!) out) -> Tensor(a!)");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_attention_v2.out", TORCH_FN(atb::_npu_paged_attention_v2));
}
}

} // namespace
