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

#include "op_plugin/utils/KernelNpuOutputDtype.h"

namespace op_infer {

at::ScalarType angle_out_dtype(const at::Tensor& self)
{
    auto out_dtype = self.scalar_type();
    if (self.is_complex()) {
        out_dtype = self.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    } else if (at::isIntegralType(out_dtype, true)) {
        out_dtype = at::kFloat;
    }
    return out_dtype;
}

at::ScalarType polar_out_dtype(const at::Tensor& abs, const at::Tensor& angle)
{
    at::ScalarType high_type = at::native::result_type(abs, angle);
    if (high_type == at::ScalarType::Float) {
        high_type = at::ScalarType::ComplexFloat;
    } else if (high_type == at::ScalarType::Double) {
        high_type = at::ScalarType::ComplexDouble;
    } else if (high_type == at::ScalarType::Half) {
        high_type = at::ScalarType::ComplexHalf;
    }
    return high_type;
}


} // namespace op_infer
