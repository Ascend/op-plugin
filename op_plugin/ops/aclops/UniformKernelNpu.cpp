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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;

namespace {
at::Tensor& uniform_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    double from,
    double to,
    c10::optional<at::Generator> generator)
{
    auto gen = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(generator,
                                                                      at_npu::detail::getDefaultNPUGenerator());
    auto pair = gen->philox_engine_inputs(10);
    const int64_t seed = static_cast<int64_t>(pair.first);
    const int64_t offset = static_cast<int64_t>(pair.second);
    c10::SmallVector<int64_t, N> seed_list = {seed};
    c10::SmallVector<int64_t, N> offset_list = {0, offset};
    int64_t alg = 1;
    at_npu::native::OpCommand cmd;
    cmd.Name("StatelessRandomUniformV2")
        .Input(self.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
        .Input(seed_list,
            at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, static_cast<string>("uint64"))
        .Input(offset_list,
            at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT, static_cast<string>("uint64"))
        .Input(at::Scalar(alg), at::ScalarType::Int)
        .Output(result)
        .Attr("dtype", self.scalar_type())
        .Run();
    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    auto tmp = result.mul(from).sub(from);
    result.mul_(to).sub_(tmp);
    return result;
}
} // namespace

at::Tensor& uniform_(at::Tensor& self, double from, double to, c10::optional<at::Generator> generator)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        uniform_out_npu(contiguous_self, contiguous_self, from, to, generator);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        uniform_out_npu(self, self, from, to, generator);
    }
    return self;
}
} // namespace acl_op
