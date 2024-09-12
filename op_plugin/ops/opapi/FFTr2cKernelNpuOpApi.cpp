// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
using npu_preparation = at_npu::native::OpPreparation;
constexpr int64_t N_FLOATS_IN_COMPLEX = 2;

at::Tensor fft_r2c_backward(
    const at::Tensor& grad,
    at::IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    int64_t last_dim_size)
{
    if (!onesided) {
        return at::real(at::_fft_c2c(grad, dim, normalization, false));
    }
    auto half_sizes = grad.sym_sizes();
    std::vector<c10::SymInt> new_grad_shape(half_sizes.begin(), half_sizes.end());
    const auto last_dim =
        at::maybe_wrap_dim(dim.back(), static_cast<int64_t>(half_sizes.size()));
    new_grad_shape[last_dim] = last_dim_size;

    const auto zero_length = last_dim_size - grad.sym_size(dim.back());

    auto gradAsReal = at::view_as_real(grad);
    new_grad_shape.push_back(N_FLOATS_IN_COMPLEX);
    auto complex_full_grad =
        zero_length > 0 ? gradAsReal.new_zeros_symint(new_grad_shape) : gradAsReal;
    if (zero_length > 0) {
      complex_full_grad.slice_symint(last_dim, 0, half_sizes[last_dim])
          .copy_(gradAsReal);
    }
    return at::real(
        at::_fft_c2c(at::view_as_complex(complex_full_grad), dim, normalization, false));
}

#endif

}  // namespace op_api
