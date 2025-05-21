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
using ElewiseParam = atb::infer::ElewiseParam;

void _npu_quantize_per_tensor(const at::Tensor &x, const at::Tensor &scale, const at::Tensor &zero_point, at::Tensor &y)
{
    const c10::OptionalDeviceGuard device_guard(device_of(x));
    OpParamCache<ElewiseParam>& elewiseParamCache = OpParamCache<ElewiseParam>::getInstance();
    ElewiseParam elewiseparam;

    elewiseparam.elewiseType = ElewiseParam::ELEWISE_QUANT_PER_CHANNEL;
    elewiseparam.quantParam.inputScale = 1.0;
    elewiseparam.quantParam.asymmetric = false;
    elewiseparam.quantParam.inputOffset = 0;
    elewiseparam.mulsParam.varAttr = 0.0;
    elewiseparam.outTensorType = ACL_DT_UNDEFINED;
    ParamSetter parametter;
    parametter.Input(x, true)
                .Input(scale, true)
                .Input(zero_point, true)
                .Output(y);
    auto opReshape = elewiseParamCache.getOperation(elewiseparam, "ElewiseCacheOperation");
    RunAtbCmd(opReshape, parametter, "ElewiseCacheOperation");
    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_quantize_per_tensor(Tensor x, Tensor scale, Tensor zero_point, Tensor(a!) y) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_quantize_per_tensor", TORCH_FN(atb::_npu_quantize_per_tensor));
}
}

} // namespace