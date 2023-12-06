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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline c10::SmallVector<int64_t, N> swiglu_backward_infershape(const at::Tensor &x, int64_t dim)
{
    if (dim < 0) {
        dim += x.sizes().size();
    }
    TORCH_CHECK(dim < x.sizes().size(), "dim out of range", dim);
    auto output_sizes = op_infer::array_to_small_vector(x.sizes());
    output_sizes[dim] /= 2;
    return output_sizes;
}

at::Tensor npu_swiglu(const at::Tensor& x, int64_t dim)
{
    auto output_sizes = swiglu_backward_infershape(x, dim);
    at::Tensor result = npu_preparation::apply_tensor_without_format(x, output_sizes);
    EXEC_NPU_CMD(aclnnSwiGlu, x, dim, result);
    return result;
}

at::Tensor npu_swiglu_backward(const at::Tensor& grad_output, const at::Tensor& x, int64_t dim)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(x);
    EXEC_NPU_CMD(aclnnSwiGluGrad, grad_output, x, dim, result);
    return result;
}

} // namespace op_api
