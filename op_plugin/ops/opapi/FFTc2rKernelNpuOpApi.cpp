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

at::Tensor fft_c2r_backward(
    const at::Tensor& grad,
    at::IntArrayRef dim,
    int64_t normalization)
{
    auto gI = at::_fft_r2c(grad, dim, normalization, true);
    auto double_length = grad.sym_size(dim.back()) - gI.sym_size(dim.back());
    if (double_length > 0) { // also covers case when signal size is zero
        auto gI_ = gI.narrow_symint(dim.back(), 1, double_length);
        gI_ = at::view_as_real(gI_);
        gI_.mul_(2);
    }
    return gI;
}

#endif

}  // namespace op_api
