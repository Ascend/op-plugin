// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

#include "op_plugin/utils/Version.h"

#if VERSION_BETWEEN(V2R11, VERSION_NEWEST)

#include <ATen/Functions.h>
#include <ATen/native/BinaryOps.h>

namespace at {
namespace native {
namespace {
at::Tensor _pow2(const at::Tensor& self, const at::Tensor& exponent)
{
    const auto self_dtype = self.scalar_type();
    if (at::isIntegralType(self_dtype, true) || self_dtype == at::kFloat) {
        return at::pow(2.0, exponent);
    }
    return at::full({}, 2.0, self.options()).pow(exponent);
}
} // namespace

void ldexp_kernel_npu(at::TensorIteratorBase& iter)
{
    const auto& self = iter.input(0);
    const auto& exponent = iter.input(1);
    auto out = iter.output();
    at::mul_out(out, self, _pow2(self, exponent));
}

REGISTER_PRIVATEUSE1_DISPATCH(ldexp_stub, &ldexp_kernel_npu);
} // namespace native
} // namespace at

#endif
