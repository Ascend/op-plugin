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
using SelfAttentionParam = atb::infer::SelfAttentionParam;
at::Tensor& _npu_flash_attention_prefix_v2(
    const at::Tensor &query,
    const at::Tensor &key_cache,
    const at::Tensor &value_cache,
    const at::Tensor &block_table,
    const at::Tensor &mask,
    c10::SymIntArrayRef seq_len,
    c10::SymIntArrayRef context_lens,
    const c10::optional<at::Tensor> &slopes,
    const int64_t kernel_type,
    const int64_t mask_type,
    int64_t num_kv_heads,
    int64_t num_heads,
    double scale_value,
    at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();
    SelfAttentionParam selfattentionparam;
    selfattentionparam.calcType = SelfAttentionParam::PREFIX_ENCODER;
    auto kerneltype = static_cast<SelfAttentionParam::KernelType>(kernel_type);
    selfattentionparam.kernelType = kerneltype;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    auto masktype = static_cast<SelfAttentionParam::MaskType>(mask_type);
    selfattentionparam.maskType = masktype;
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
    at::Tensor seq_len_tensor = at::tensor(c10::asIntArrayRefUnchecked(seq_len), at::kInt);
    at::Tensor context_lens_tensor = at::tensor(c10::asIntArrayRefUnchecked(context_lens), at::kInt);
    if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_NORM_COMPRESS) {
        parametter.Input(query)
            .Input(key_cache)
            .Input(value_cache)
            .Input(block_table)
            .Input(mask)
            .Input(seq_len_tensor)
            .Input(context_lens_tensor)
            .Output(out);
    } else if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
              selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT) {
        parametter.Input(query)
            .Input(key_cache)
            .Input(value_cache)
            .Input(block_table)
            .Input(mask)
            .Input(seq_len_tensor)
            .Input(context_lens_tensor)
            .Input(slopes)
            .Output(out);
    }
    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");

    return out;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention_prefix_v2.out(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor mask, SymInt[] seq_len, SymInt[] context_lens, *, Tensor? slopes=None, int kernel_type=1, int mask_type=3, int num_kv_heads=0, int num_heads=0, float scale_value=1, Tensor(a!) out) -> Tensor(a!)");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention_prefix_v2.out", TORCH_FN(atb::_npu_flash_attention_prefix_v2));
}
}
} // namespace
