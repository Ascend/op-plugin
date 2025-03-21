// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#include <ATen/NamedTensorUtils.h>
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
    using npu_preparation = at_npu::native::OpPreparation;
    using npu_compile_type = at_npu::native::CompileType;
    using npu_utils = at_npu::native::NpuUtils;

namespace {
    at::Tensor &bernoulli_npu_nocheck(at::Tensor &result, double p, c10::optional<at::Generator> gen)
    {
        auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(
            gen,
            at_npu::detail::getDefaultNPUGenerator());
        auto pair = gen_->philox_engine_inputs(10);
        const int64_t seed = static_cast<int64_t>(pair.first);
        const int64_t offset = static_cast<int64_t>(pair.second);

        at_npu::native::OpCommand cmd;
        cmd.Name("StatelessBernoulli")
            .Input(result.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(at::Scalar(p), at::kFloat)
            .Input(at::Scalar(seed), at::kLong)
            .Input(at::Scalar(offset), at::kLong)
            .Output(result)
            .Attr("dtype", result.scalar_type())
            .Run();
        return result;
    }

    at::Tensor &bernoulli_npu_nocheck(at::Tensor &result, const at::Tensor &p, c10::optional<at::Generator> gen)
    {
        auto gen_ = at::get_generator_or_default<at_npu::NPUGeneratorImpl>(
            gen,
            at_npu::detail::getDefaultNPUGenerator());
        auto pair = gen_->philox_engine_inputs(10);
        const int64_t seed = static_cast<int64_t>(pair.first);
        const int64_t offset = static_cast<int64_t>(pair.second);

        at_npu::native::OpCommand cmd;
        cmd.Name("StatelessBernoulli")
            .Input(result.sizes(), at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(p)
            .Input(at::Scalar(seed), at::kLong)
            .Input(at::Scalar(offset), at::kLong)
            .Output(result)
            .Attr("dtype", result.scalar_type())
            .Run();
        return result;
    }
} // namespace

at::Tensor &bernoulli_(at::Tensor &self, double p, c10::optional<at::Generator> gen)
{
    if (!self.is_contiguous()) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        bernoulli_npu_nocheck(contiguous_self, p, gen);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        bernoulli_npu_nocheck(self, p, gen);
    }
    return self;
}

at::Tensor &bernoulli_(at::Tensor &self, const at::Tensor &p, c10::optional<at::Generator> gen)
{
    at::Tensor p_ori_format = npu_preparation::CastBackToOriFormat(p);
    npu_preparation::CheckMemory({self, p}, {self});

    if (!self.is_contiguous()) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        bernoulli_npu_nocheck(contiguous_self, p_ori_format, gen);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        bernoulli_npu_nocheck(self, p_ori_format, gen);
    }
    return self;
}

at::Tensor bernoulli(const at::Tensor &self, c10::optional<at::Generator> gen)
{
    at::Tensor result = npu_preparation::apply_tensor_with_format(self.sizes(), self.options(), ACL_FORMAT_ND);
    bernoulli_npu_nocheck(result, self, gen);
    return result;
}

at::Tensor bernoulli(const at::Tensor &self, double p, c10::optional<at::Generator> gen)
{
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT).bernoulli_(p, gen);
}

at::Tensor &bernoulli_out(const at::Tensor &self, c10::optional<at::Generator> gen, at::Tensor &result)
{
    result.resize_(self.sizes()).bernoulli_(self, gen);
    at::namedinference::propagate_names(result, self);
    return result;
}
} // namespace acl_op
