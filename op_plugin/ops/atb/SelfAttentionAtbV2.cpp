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
at::Tensor& _npu_flash_attention_v2(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    c10::SymIntArrayRef seq_len,
    const c10::optional<at::Tensor> &mask,
    const c10::optional<at::Tensor> &slopes,
    int64_t kernel_type,
    int64_t mask_type,
    double scale_value,
    int64_t num_heads,
    int64_t num_kv_heads,
    at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();

    SelfAttentionParam selfattentionparam;
    selfattentionparam.calcType = SelfAttentionParam::PA_ENCODER;
    auto kerneltype = static_cast<SelfAttentionParam::KernelType>(kernel_type);
    selfattentionparam.kernelType = kerneltype;
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    auto masktype = static_cast<SelfAttentionParam::MaskType>(mask_type);
    selfattentionparam.maskType = masktype;
    selfattentionparam.kvcacheCfg = SelfAttentionParam::K_CACHE_V_CACHE;
    selfattentionparam.scaleType = SelfAttentionParam::SCALE_TYPE_TOR;
    selfattentionparam.quantType = SelfAttentionParam::TYPE_QUANT_UNDEFINED;
    selfattentionparam.cacheType = SelfAttentionParam::CACHE_TYPE_NORM;
    selfattentionparam.outDataType = ACL_DT_UNDEFINED;
    selfattentionparam.headNum = num_heads;
    selfattentionparam.kvHeadNum = num_kv_heads;
    selfattentionparam.qScale = 1;
    selfattentionparam.qkScale = scale_value;
    selfattentionparam.batchRunStatusEnable = false;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;

    if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI ||
        selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
        selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
        selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
        selfattentionparam.isTriuMask = 1;
    } else if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_NORM) {
        selfattentionparam.isTriuMask = 0;
    }

    ParamSetter parametter;
    at::Tensor seq_len_tensor = at::tensor(c10::asIntArrayRefUnchecked(seq_len), at::kInt);
    if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS ||
        selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_SQRT ||
        selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN) {
        parametter.Input(query)
            .Input(key)
            .Input(value)
            .Input(mask)
            .Input(seq_len_tensor)
            .Input(slopes)
            .Output(out);
    } else if (selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_ALIBI || selfattentionparam.maskType == SelfAttentionParam::MASK_TYPE_NORM) {
        parametter.Input(query)
            .Input(key)
            .Input(value)
            .Input(mask)
            .Input(seq_len_tensor)
            .Output(out);
    }
    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");
    return out;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention_v2.out(Tensor query, Tensor key, Tensor value, SymInt[] seq_len, *, Tensor? mask=None, Tensor? slopes=None, int kernel_type=0, int mask_type=2, float scale_value=1, int num_heads=0, int num_kv_heads=0, Tensor(a!) out) -> Tensor(a!)");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention_v2.out", TORCH_FN(atb::_npu_flash_attention_v2));
}
}
} // namespace
