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
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/AclOpsInterface.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace op_api {

at::Tensor& npu_sim_exponential_(at::Tensor& self, double lambd, c10::optional<at::Generator> generator)
{
    TORCH_CHECK(lambd > 0.0, "npu_sim_exponential_ expects lambd > 0.0, but found lambd=",
        lambd, OPS_ERROR(ErrCode::PARAM));
    if (std::isinf(lambd)) {
        self.zero_();
        return self;
    }

    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    int64_t seed = static_cast<int64_t>(pair.first);
    int64_t offset = static_cast<int64_t>(pair.second);
    int64_t count = self.numel();
    ASCEND_LOGI("count:%lld, lambd:%lf, seed:%lld, offset:%lld", count, lambd, seed, offset);

    EXEC_NPU_CMD(aclnnSimThreadExponential, self, count, lambd, seed, offset);
    return self;
}

}   // namespace op_api