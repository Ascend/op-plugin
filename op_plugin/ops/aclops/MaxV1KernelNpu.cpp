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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
std::tuple<at::Tensor&, at::Tensor&> max_v1_out_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  at_npu::native::OpCommand cmd;
  cmd.Name("ArgMaxWithValue")
      .Input(self)
      .Output(indices)
      .Output(output)
      .Attr("dimension", dim)
      .Attr("keep_dims", keepdim)
      .Run();
  return std::tie(output, indices);
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, int64_t dim, bool keepdim) {
  c10::SmallVector<int64_t, SIZE> dims = {dim};
  c10::SmallVector<int64_t, SIZE> output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  c10::SmallVector<int64_t, SIZE> indices_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
  int64_t npu_format = output_size.empty() ? ACL_FORMAT_NCHW : npu_preparation::get_tensor_npu_format(self);

  at::Tensor outputs = npu_preparation::apply_tensor_with_format(
      output_size, self.options(), npu_format);
  at::Tensor indices = npu_preparation::apply_tensor_with_format(
      indices_size, self.options().dtype(at::kInt), ACL_FORMAT_NCHW);
  max_v1_out_nocheck(outputs, indices, self, dim, keepdim);
  return std::tie(outputs, indices);
}

std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, at::Dimname dim, bool keepdim) {
  return acl_op::npu_max(self, dimname_to_position(self, dim), keepdim);
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor npu_max_backward(
    const at::Tensor& grad,
    int64_t dim,
    const at::Tensor& indices,
    at::IntArrayRef sizes,
    bool keepdim) {
    at::Tensor new_grad = grad;
    at::Tensor new_indices = indices;
    if (keepdim && sizes.size() > 0) {
        new_grad = grad.squeeze(dim);
        new_indices = indices.squeeze(dim);
    }
    if (new_indices.dtype() == at::kLong) {
        new_indices = at_npu::native::custom_ops::npu_dtype_cast(new_indices, at::kInt);
    }
    auto grad_input = acl_op::npu_scatter(at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
    return grad_input;
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor npu_max_backward_symint(const at::Tensor &grad, int64_t dim, const at::Tensor &indices,
                                   c10::SymIntArrayRef sizes_symint, bool keepdim)
{
    at::IntArrayRef sizes = c10::asIntArrayRefUnchecked(sizes_symint);
    at::Tensor new_grad = grad;
    at::Tensor new_indices = indices;
    if (keepdim && sizes.size() > 0) {
        new_grad = grad.squeeze(dim);
        new_indices = indices.squeeze(dim);
    }
    if (new_indices.dtype() == at::kLong) {
        new_indices = at_npu::native::custom_ops::npu_dtype_cast(new_indices, at::kInt);
    }
    auto grad_input = acl_op::npu_scatter(at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
    return grad_input;
}
#endif
} // namespace acl_op
