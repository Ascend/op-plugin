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

#include "op_plugin/ops/OpInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_plugin {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& addr_out(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  npu_utils::check_1d(vec1, "vec1", "addr");
  npu_utils::check_1d(vec2, "vec2", "addr");
  at::Tensor mat1 = vec1.unsqueeze(1);
  at::Tensor mat2 = vec2.unsqueeze(0);
  at::Tensor mm_result = at::mm(mat1, mat2);
  at::Tensor mm_mul_result = at::mul(mm_result, alpha);

  at::add_out(result, mm_mul_result, self, beta);
  return result;
}

at::Tensor addr(
    const at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto output_size = op_infer::addr_npu_output_size(self, vec1, vec2, beta, alpha);
  at::Tensor result = npu_preparation::ApplyTensor(self, output_size);
  addr_out(self, vec1, vec2, beta, alpha, result);
  return result;
}

at::Tensor& addr_(
    at::Tensor& self,
    const at::Tensor& vec1,
    const at::Tensor& vec2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  npu_preparation::CheckMemory({self, vec1, vec2}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    op_plugin::addr_out(contiguous_self, vec1, vec2, beta, alpha, contiguous_self);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    op_plugin::addr_out(self, vec1, vec2, beta, alpha, self);
  }

  return self;
}
} // namespace op_plugin
