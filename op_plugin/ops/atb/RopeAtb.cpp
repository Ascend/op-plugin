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
    using RopeParam = atb::infer::RopeParam;

    static at::Tensor sequenceLength;
    static int64_t previousTokenCount = -1;

    static at::Tensor cosCacheNeox;
    static at::Tensor sinCacheNeox;
    static at::Tensor cosCache;
    static at::Tensor sinCache;

    void InitializeCosSinCache(const at::Tensor &cos_sin_cache)
    {
        auto cosSinChunks = cos_sin_cache.chunk(2, -1);

        cosCache = cosSinChunks[0].repeat_interleave(2, 1);
        sinCache = cosSinChunks[1].repeat_interleave(2, 1);
        cosCacheNeox = cosSinChunks[0].repeat({1, 2});
        sinCacheNeox = cosSinChunks[1].repeat({1, 2});
    }

    void _npu_rotary_embedding(const at::Tensor &positions, at::Tensor &query, at::Tensor &key, int64_t head_size, const at::Tensor &cos_sin_cache, bool is_neox_style)
    {
        const c10::OptionalDeviceGuard device_guard(device_of(positions));
        if (!cosCache.defined() || !sinCache.defined()) {
            InitializeCosSinCache(cos_sin_cache);
        }

        at::Tensor flatPositions = positions.flatten();
        int32_t currentTokenCount = flatPositions.size(0);

        at::Tensor cos = is_neox_style ? cosCacheNeox.index_select(0, flatPositions)
                                    : cosCache.index_select(0, flatPositions);
        at::Tensor sin = is_neox_style ? sinCacheNeox.index_select(0, flatPositions)
                                    : sinCache.index_select(0, flatPositions);

        if (!sequenceLength.defined() || previousTokenCount != currentTokenCount) {
            previousTokenCount = currentTokenCount;
            sequenceLength = at::tensor({currentTokenCount}, at::kInt).to(query.device());
        }

        RopeParam ropeparam;
        ropeparam.rotaryCoeff = is_neox_style ? 2 : head_size;

        ParamSetter parametter;
        parametter.Input(query, true)
            .Input(key, true)
            .Input(cos, true)
            .Input(sin, true)
            .Input(sequenceLength, true)
            .Output(query)
            .Output(key);

        OpParamCache<RopeParam> &ropeParamCache = OpParamCache<RopeParam>::getInstance();
        auto opRope = ropeParamCache.getOperation(ropeparam, "RopeOperation");
        RunAtbCmd(opRope, parametter, "RopeOperation");
    }

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!) key, int head_size, Tensor cos_sin_cache, bool is_neox_style) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_rotary_embedding", TORCH_FN(atb::_npu_rotary_embedding));
}
}

} // namespace
