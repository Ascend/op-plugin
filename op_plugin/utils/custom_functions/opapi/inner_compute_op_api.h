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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE_OP_API__
#define __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE_OP_API__

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

namespace op_api {
at::Tensor& sum_out_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                                   c10::optional<c10::ScalarType> dtype, at::Tensor& result);
at::Tensor sum_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                              c10::optional<c10::ScalarType> dtype);
std::tuple<at::Tensor, at::Tensor> _fused_moving_avg_obs_fq_helper_common(
    const at::Tensor& self, const at::Tensor& observer_on, const at::Tensor& fake_quant_on,
    at::Tensor& running_min, at::Tensor& running_max, at::Tensor& scale, at::Tensor& zero_point,
    const double averaging_const, const int64_t quant_min, const int64_t quant_max, const int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant);
at::Tensor matmul_mat1_backward(const at::Tensor self, const at::Tensor other, const at::Tensor grad_output);
at::Tensor matmul_mat2_backward(const at::Tensor self, const at::Tensor other, const at::Tensor grad_output);
} // namespace op_api
#endif
