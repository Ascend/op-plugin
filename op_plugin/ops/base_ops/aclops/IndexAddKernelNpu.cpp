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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& index_add_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  at::Tensor indices = index;
  if (index.scalar_type() != at::ScalarType::Int) {
    indices = at_npu::native::custom_ops::npu_dtype_cast(index, at::kInt);
  }
  if (index.dim() == 0) {
    indices.unsqueeze_(0);
  }

  at::SmallVector<int64_t, N> pad_size = op_infer::array_to_small_vector(self.sizes());
  pad_size[dim] = indices.sizes()[0];
  at::Tensor source_broadcast = acl_op::npu_broadcast(source, pad_size);
  source_broadcast.mul_(alpha);
  at_npu::native::OpCommand cmd;
  cmd.Name("InplaceIndexAdd")
      .Input(self)
      .Input(indices)
      .Input(source_broadcast)
      .Output(result)
      .Attr("axis", dim)
      .Run();
  return result;
}
} // namespace

at::Tensor& index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& result) {
  npu_preparation::CheckOut(
      {self, index, source},
      result,
      self);
  if (!npu_utils::check_match(&result)) {
    at::Tensor contiguous_result = npu_utils::format_contiguous(result);
    index_add_out_npu_nocheck(contiguous_result, self, dim, index, source, alpha);
    npu_utils::format_fresh_view(result, contiguous_result);
  } else {
    index_add_out_npu_nocheck(result, self, dim, index, source, alpha);
  }
  return result;
}

at::Tensor index_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  at::Tensor result(self.clone());
  index_add_out_npu_nocheck(result, result, dim, index, source, alpha);
  return result; 
}

at::Tensor index_add(
    const at::Tensor& self,
    at::Dimname dim, 
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha) {
  return acl_op::index_add(self, dimname_to_position(self, dim), index, source, alpha);
}
} // namespace acl_op
