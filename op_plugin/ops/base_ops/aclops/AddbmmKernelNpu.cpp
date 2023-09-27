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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& addbmm_out(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
  at::Tensor mul_result = at::mul(batch1, alpha);
  at::Tensor bmm_result = at::bmm(mul_result, batch2);
  int64_t dim[2] = {batch1.size(1), batch2.size(2)};
  at::Tensor sum_result = at::sum_to(bmm_result, dim);
  // sum_result + self*beta
  at::add_out(result, sum_result, self, beta);
  return result;
}

at::Tensor addbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  auto output_size = op_infer::addbmm_npu_output_size(self, batch1, batch2, beta, alpha);
  at::Tensor result = npu_preparation::apply_tensor(self, output_size);
  acl_op::addbmm_out(self, batch1, batch2, beta, alpha, result);
  return result;
}

at::Tensor& addbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  npu_preparation::CheckMemory({self, batch1, batch2}, {self});
  if (!npu_utils::check_match(&self)) {
    at::Tensor contiguous_self = npu_utils::format_contiguous(self);
    acl_op::addbmm_out(contiguous_self, batch1, batch2, beta, alpha, contiguous_self);
    npu_utils::format_fresh_view(self, contiguous_self);
  } else {
    acl_op::addbmm_out(self, batch1, batch2, beta, alpha, self);
  }
  return self;
}
} // namespace acl_op
