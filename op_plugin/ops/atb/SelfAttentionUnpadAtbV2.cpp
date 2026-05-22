// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
void _npu_flash_attention_unpad_v2(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &seq_len, const double scale_value, const int64_t num_heads, const int64_t num_kv_heads, const c10::optional<int64_t> kernel_type, at::Tensor &out)
{
    const c10::OptionalDeviceGuard device_guard(device_of(query));
    int64_t ktype = kernel_type.has_value() ? kernel_type.value() : 0;
    TORCH_CHECK(ktype == SelfAttentionParam::KERNELTYPE_DEFAULT ||
                ktype == SelfAttentionParam::KERNELTYPE_EXP_M8V2,
                "kernel_type only supports KERNELTYPE_DEFAULT(0) and KERNELTYPE_EXP_M8V2(2), but got: ", ktype);
    OpParamCache<SelfAttentionParam>& selfAttentionParamCache = OpParamCache<SelfAttentionParam>::getInstance();
    SelfAttentionParam selfattentionparam;

    selfattentionparam.calcType = SelfAttentionParam::PA_ENCODER;
    selfattentionparam.kernelType = static_cast<SelfAttentionParam::KernelType>(ktype);
    selfattentionparam.clampType = SelfAttentionParam::CLAMP_TYPE_UNDEFINED;
    selfattentionparam.maskType = SelfAttentionParam::MASK_TYPE_UNDEFINED;
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
    selfattentionparam.isTriuMask = 0;
    selfattentionparam.clampMin = 0;
    selfattentionparam.clampMax = 0;
    selfattentionparam.inputLayout = atb::infer::TYPE_BSND;
    selfattentionparam.mlaVHeadSize = 0;
    selfattentionparam.windowSize = 0;
    ParamSetter parametter;
    parametter.Input(query, true)
        .Input(key, true)
        .Input(value, true)
        .Input(seq_len, true)
        .Output(out);

    auto opSelfattention = selfAttentionParamCache.getOperation(selfattentionparam, "SelfAttentionOperation");
    RunAtbCmd(opSelfattention, parametter, "SelfAttentionOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_flash_attention_unpad_v2(Tensor query, Tensor key, Tensor value, Tensor seq_len, float scale_value, int num_heads, int num_kv_heads, *, int ? kernel_type = 0, Tensor(a!) out) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_flash_attention_unpad_v2", TORCH_FN(atb::_npu_flash_attention_unpad_v2));
}
}

} // namespace