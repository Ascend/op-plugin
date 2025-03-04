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

using LinearParam = atb::infer::LinearParam;
void _npu_matmul_add_fp32(const at::Tensor &x, const at::Tensor &weight, at::Tensor & C)
{
    const c10::OptionalDeviceGuard device_guard(device_of(x));
    OpParamCache<LinearParam>& linearParamCache = OpParamCache<LinearParam>::getInstance();
    LinearParam  linearParam;
    linearParam.transposeA = true;                    // 是否转置A矩阵
    linearParam.transposeB = false;                     // 是否转置B矩阵
    linearParam.hasBias = false;
    linearParam.enAccum = true;

    auto opLinear = linearParamCache.getOperation(linearParam, "LinearOperation");
    ParamSetter paramsetter;
    paramsetter.Input(x)
                .Input(weight)
                .Input(C)
                .Output(C);

    RunAtbCmd(opLinear, paramsetter, "LinearOperation");
    return ;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_matmul_add_fp32(Tensor x, Tensor weight, Tensor(a!) C) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_matmul_add_fp32", TORCH_FN(atb::_npu_matmul_add_fp32));
}
}

} // namespace atb
