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

#include <climits>
#include <ATen/NamedTensorUtils.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& bernoulli_(at::Tensor& self, double p, c10::optional<at::Generator> gen)
{
    return op_api::le_(op_api::uniform_(self, 0.0, 1.0, gen), at::Scalar(p));
}

at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p, c10::optional<at::Generator> gen)
{
    return op_api::le_(op_api::uniform_(self, 0.0, 1.0, gen), p);
}

at::Tensor bernoulli(const at::Tensor& self, c10::optional<at::Generator> gen)
{
    at::Tensor self_copy = npu_preparation::apply_tensor_without_format(self);
    return op_api::le_(op_api::uniform_(self_copy, 0.0, 1.0, gen), self);
}

at::Tensor bernoulli(const at::Tensor& self, double p, c10::optional<at::Generator> gen)
{
    return at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT).bernoulli_(p, gen);
}

at::Tensor& bernoulli_out(const at::Tensor& self, c10::optional<at::Generator> gen, at::Tensor& result)
{
    result.resize_(self.sizes()).bernoulli_(self, gen);
    at::namedinference::propagate_names(result, self);
    return result;
}
}  // namespace op_api
