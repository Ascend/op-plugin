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

using RmsNormParam = atb::infer::RmsNormParam;
void _npu_quant_rms_norm(const at::Tensor &x,
                         const at::Tensor &gamma,
                         const at::Tensor &beta,
                         const at::Tensor &scale,
                         const at::Tensor &offset,
                         at::Tensor & output,
                         double eps)
{
    OpParamCache<RmsNormParam>& rmsnormParamCache = OpParamCache<RmsNormParam>::getInstance();
    RmsNormParam  rmsnormParam;
    rmsnormParam.layerType = atb::infer::RmsNormParam::RMS_NORM_NORM;
    rmsnormParam.normParam.quantType = atb::infer::QUANT_INT8;
    rmsnormParam.normParam.epsilon = eps;

    ParamSetter paramsetter;
    paramsetter.Input(x, true)
                .Input(gamma, true)
                .Input(beta, true)
                .Input(scale, true)
                .Input(offset, true)
                .Output(output);

    auto opRmsNorm = rmsnormParamCache.getOperation(rmsnormParam, "QuantRmsNormOperation");
    RunAtbCmd(opRmsNorm, paramsetter, "QuantRmsNormOperation");
    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_quant_rms_norm(Tensor self, Tensor gamma, Tensor beta, Tensor scale, Tensor offset, Tensor(a!) output, float eps=1e-05) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_quant_rms_norm", TORCH_FN(atb::_npu_quant_rms_norm));
}
}

} // namespace atb
