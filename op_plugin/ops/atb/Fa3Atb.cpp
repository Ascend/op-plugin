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
void _npu_paged_attention_quant(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &value_cache, const int64_t num_kv_heads, const int64_t num_heads, const double scale_value, const at::Tensor &block_table, const at::Tensor &context_lens,
    const int64_t quant_type, const int64_t outdata_type, const at::Tensor &k_descale, const at::Tensor &v_descale, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<PagedAttentionParam>& pagedAttentionParamCache = OpParamCache<PagedAttentionParam>::getInstance();
    PagedAttentionParam pagedparam;

    pagedparam.headNum = num_heads;
    pagedparam.qkScale = scale_value;
    pagedparam.kvHeadNum = num_kv_heads;
    pagedparam.maskType = PagedAttentionParam::UNDEFINED;
    pagedparam.batchRunStatusEnable = false;

    auto quanttype = static_cast<PagedAttentionParam::QuantType>(quant_type);
    pagedparam.quantType = quanttype;
    auto outdataType = static_cast<aclDataType>(outdata_type);
    pagedparam.outDataType = outdataType;
    pagedparam.hasQuantOffset = false;
    pagedparam.compressType = PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    pagedparam.calcType = PagedAttentionParam::CALC_TYPE_UNDEFINED;
    pagedparam.scaleType = PagedAttentionParam::SCALE_TYPE_TOR;
    pagedparam.inputLayout = atb::infer::TYPE_BSND;
    pagedparam.mlaVHeadSize = 0;

    ParamSetter paramsetter;
    paramsetter.Input(query, true)
            .Input(key_cache)
            .Input(value_cache)
            .Input(block_table, true)
            .Input(context_lens, true)
            .Input(k_descale, true)
            .Input(v_descale, true)
            .Output(out);
    auto opPaged = pagedAttentionParamCache.getOperation(pagedparam, "PagedAttentionOperation");
    RunAtbCmd(opPaged, paramsetter, "PagedAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_paged_attention_quant(Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, int num_heads, float scale_value, Tensor block_table, Tensor context_lens, int quant_type, int outdata_type, Tensor k_descale, Tensor v_descale, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_paged_attention_quant", TORCH_FN(atb::_npu_paged_attention_quant));
}
}

}