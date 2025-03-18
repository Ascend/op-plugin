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

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
namespace {
at::Tensor int_repr_quantized_nocheck(const at::Tensor& self)
{
    auto dtype = self.scalar_type();
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    }
    at::Tensor tmp = at::empty(
        self.sizes(),
        self.options().dtype(output_dtype).memory_format(self.suggest_memory_format()));
    at::Tensor result = at::empty(
        self.sizes(),
        self.options().dtype(output_dtype).memory_format(self.suggest_memory_format()));

    at_npu::native::NPUNativeFunctions::set_(tmp, self);
    result = at_npu::native::custom_ops::npu_dtype_cast(tmp, output_dtype);

    return result;
}
} // namespace

at::Tensor int_repr(const at::Tensor& self)
{
    return int_repr_quantized_nocheck(self);
}
#endif
} // namespace acl_op
