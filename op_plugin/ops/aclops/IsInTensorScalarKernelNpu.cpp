// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor isin(const at::Tensor &elements, const at::Scalar &test_elements, bool assume_unique, bool invert)
{
    const auto elements_cpu = elements.cpu();
    at::Tensor result = at::isin(elements_cpu, test_elements, assume_unique, invert);
    result = result.to(elements.device());
    return result;
}

at::Tensor &isin_out(const at::Tensor &elements, const at::Scalar &test_elements, bool assume_unique, bool invert,
                     at::Tensor &result)
{
    npu_preparation::CheckOut({elements}, result, npu_preparation::get_tensor_npu_format(elements),
                              at::ScalarType::Bool, elements.sizes());
    const auto elements_cpu = elements.cpu();
    auto result_cpu = result.cpu();
    at::isin_out(result_cpu, elements_cpu, test_elements, assume_unique, invert);
    result.copy_(result_cpu);
    return result;
}
} // namespace acl_op
