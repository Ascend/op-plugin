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

using GroupTopkParam = atb::infer::GroupTopkParam;
void _npu_group_topk(const at::Tensor &self, int64_t k, int64_t group_num, int64_t n)
{
    const c10::OptionalDeviceGuard device_guard(device_of(self));
    OpParamCache<GroupTopkParam>& GroupTopkParamCache = OpParamCache<GroupTopkParam>::getInstance();
    GroupTopkParam GroupTopkParam;
    GroupTopkParam.groupNum = static_cast<int32_t>(group_num);
    GroupTopkParam.k = static_cast<int32_t>(k);
    GroupTopkParam.n = static_cast<uint16_t>(n);
    GroupTopkParam.groupMultiFlag = static_cast<GroupTopkParam::GroupMultiFlag>(0);
    if (n > 1) {
        GroupTopkParam.groupMultiFlag = static_cast<GroupTopkParam::GroupMultiFlag>(1);
    }

    at::Tensor out = self;
    auto idx = at::arange(1024, self.options().device(at::Device(at::kPrivateUse1)).dtype(at::kInt));

    ParamSetter paramsetter;
    paramsetter.Input(self, true)
                .Input(idx, true)
                .Output(out);
    auto opGroupTopk = GroupTopkParamCache.getOperation(GroupTopkParam, "GroupTopkOperation");
    RunAtbCmd(opGroupTopk, paramsetter, "GroupTopkOperation");
    return;
}

namespace {
TORCH_LIBRARY_FRAGMENT(atb, m)
{
    m.def("_npu_group_topk(Tensor self, int k=0, int group_num=1, int n=1) -> ()");
}
}

namespace {
TORCH_LIBRARY_IMPL(atb, PrivateUse1, m)
{
    m.impl("_npu_group_topk", TORCH_FN(atb::_npu_group_topk));
}
}

} // namespace atb
