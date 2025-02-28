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
using ReshapeAndCacheParam = atb::infer::ReshapeAndCacheParam;
void _npu_reshape_and_cache(const at::Tensor &key, const at::Tensor &value, at::Tensor &key_cache, at::Tensor &value_cache, const at::Tensor &slot_indices)
{
    const c10::OptionalDeviceGuard device_guard(device_of(key));
    OpParamCache<ReshapeAndCacheParam>& reshapeAndCacheParamCache = OpParamCache<ReshapeAndCacheParam>::getInstance();
    ReshapeAndCacheParam reshapeparam;
    reshapeparam.compressType = ReshapeAndCacheParam::COMPRESS_TYPE_UNDEFINED;
    reshapeparam.kvCacheCfg = ReshapeAndCacheParam::K_CACHE_V_CACHE;
    
    ParamSetter parametter;
    parametter.Input(key)
        .Input(value)
        .Input(key_cache)
        .Input(value_cache)
        .Input(slot_indices)
        .Output(key_cache)
        .Output(value_cache);
    auto opReshape = reshapeAndCacheParamCache.getOperation(reshapeparam, "ReshapeCacheOperation");
    RunAtbCmd(opReshape, parametter, "ReshapeCacheOperation");

    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_reshape_and_cache(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slot_indices) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_reshape_and_cache", TORCH_FN(atb::_npu_reshape_and_cache));
}
}
} // namespace