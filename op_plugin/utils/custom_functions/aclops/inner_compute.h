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

#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE__
#define __TORCH_NPU_OP_PLUGIN_UTILS_INNER_COMPUTE__

#include <ATen/ATen.h>
#include <ATen/Tensor.h>

namespace op_plugin {
at::Tensor embedding_common_nocheck(const at::Tensor& weight, const at::Tensor& indices);
at::Tensor gelu_common_nocheck(const at::Tensor& self);
at::Tensor gelu_backward_common_nocheck(const at::Tensor& grad, const at::Tensor& self);
std::tuple<at::Tensor, at::Tensor> grid_sampler3d_backward_common_nocheck(const at::Tensor& grad, const at::Tensor& input,
                                                                   const at::Tensor& grid, int64_t interpolation_mode,
                                                                   int64_t padding_mode, bool align_corners);
at::Tensor& sum_out_common_nocheck(at::Tensor& result, const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                           c10::optional<c10::ScalarType> dtype);
at::Tensor sum_common_nocheck(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_expand_outplace(const at::Tensor& to_expand1, const at::Tensor& to_expand2, const at::Tensor& to_expand3, const char *api_name);
at::Tensor& where_out_nocheck(at::Tensor& out, const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other);
at::Tensor& index_copy_npu_impl(const int64_t dim, const at::Tensor& index, const at::Tensor& source, at::Tensor& result);
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_svd_out_common(const at::Tensor& A, const bool full_matrices,
                                                                        const bool compute_uv, at::Tensor& U,
                                                                        at::Tensor& S, at::Tensor& Vh);
}  // namespace op_plugin
#endif
