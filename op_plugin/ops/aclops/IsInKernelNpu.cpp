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

at::Tensor& isin_out(const at::Scalar& element, const at::Tensor &test_element,
                     bool assume_unique, bool invert, at::Tensor& result)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    npu_preparation::CheckOut({test_element}, result, npu_preparation::get_tensor_npu_format(test_element),
                              at::ScalarType::Bool, shape_small_vec);
    const auto test_element_cpu = test_element.cpu();
    auto result_cpu = result.cpu();
    at::isin_out(result_cpu, element, test_element_cpu, assume_unique, invert);
    result.copy_(result_cpu);
    return result;
}
} // namespace acl_op
