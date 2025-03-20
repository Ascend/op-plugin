// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& uniform_(at::Tensor& self, double from, double to,
                     c10::optional<at::Generator> gen_)
{
    DO_COMPATIBILITY(aclnnInplaceUniform, acl_op::uniform_(self, from, to, gen_));
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(gen_, at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    int64_t seed = static_cast<int64_t>(pair.first);
    int64_t offset = static_cast<int64_t>(pair.second);
    EXEC_NPU_CMD(aclnnInplaceUniform, self, from, to, seed, offset);
    return self;
}

}
