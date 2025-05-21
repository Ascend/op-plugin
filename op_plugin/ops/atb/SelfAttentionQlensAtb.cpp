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
using SelfAttentionParam = atb::infer::SelfAttentionParam;
void _npu_flash_attention_qlens(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &value_cache, const at::Tensor &block_table, const at::Tensor &mask, const at::Tensor &seq_len, const at::Tensor &context_lens, int64_t num_kv_heads, int64_t num_heads, double scale_value, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();
    SelfAttentionParam selfattentionparam;

    selfattentionparam.calcType = SelfAttentionParam::PREFIX_ENCODER;
    selfattentionparam.kernelType = SelfAttentionParam::KERNELTYPE_HIGH_PRECISION;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    selfattentionparam.maskType = SelfAttentionParam::MASK_TYPE_NORM_COMPRESS;
    selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
    selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
    selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNQUANT;
    selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
    selfattentionparam.outDataType = ACL_DT_UNDEFINED;
    selfattentionparam.headNum = num_heads;
    selfattentionparam.kvHeadNum = num_kv_heads;
    selfattentionparam.qScale = 1;
    selfattentionparam.qkScale = scale_value;
    selfattentionparam.batchRunStatusEnable = false;
    selfattentionparam.isTriuMask = 1;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;

    ParamSetter parametter;
    parametter.Input(query, true)
        .Input(key_cache)
        .Input(value_cache)
        .Input(block_table, true)
        .Input(mask)
        .Input(seq_len, true)
        .Input(context_lens, true)
        .Output(out);

    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention_qlens(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor mask, Tensor seq_len, Tensor context_lens, int num_kv_heads, int num_heads, float scale_value, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention_qlens", TORCH_FN(atb::_npu_flash_attention_qlens));
}
}
} // namespace
