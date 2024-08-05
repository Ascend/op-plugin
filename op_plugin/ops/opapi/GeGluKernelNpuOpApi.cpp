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

std::tuple<at::Tensor, at::Tensor> npu_geglu(const at::Tensor &self, int64_t dim, int64_t approximate, bool activate_left)
{
    auto dim_num = self.dim();
    if (dim_num == 0) {
        dim_num = 1;
    }
    TORCH_CHECK(-dim_num - 1 < dim && dim < dim_num, " Expected npu_swiglu dim value ", dim,
                " to be in range [", -dim_num, ", ", (dim_num - 1), "] but check failed.", OPS_ERROR(ErrCode::VALUE));
    auto output_size = op_infer::array_to_small_vector(self.sizes());
    int64_t slice_dim = dim < 0 ? dim_num + dim : dim;

    auto slice_dim_size = output_size[slice_dim];
    output_size[slice_dim] = slice_dim_size / 2;

    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    at::Tensor result_gelu = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnGeGluV3, self, dim, approximate, activate_left, result, result_gelu);
    return std::tuple<at::Tensor, at::Tensor>(result, result_gelu);
}

at::Tensor npu_geglu_grad(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& gelu, int64_t dim,
                          int64_t approximate, bool activate_left)
{
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(self);
    EXEC_NPU_CMD(aclnnGeGluV3Backward, grad_output, self, gelu, dim, approximate, activate_left, grad_input);
    return grad_input;
}

}
