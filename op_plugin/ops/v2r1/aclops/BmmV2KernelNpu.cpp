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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
at::Tensor npu_bmm_v2_mat1_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::SymIntArrayRef sizes_symint) {
  at::IntArrayRef sizes = c10::asIntArrayRefUnchecked(sizes_symint);
  // da = grad * b^T
  auto grad_with_full_size = grad;

  std::vector<int64_t> axis_reshape(grad.sizes().begin(), grad.sizes().end());
  if (mat1.dim() == 1) {
    axis_reshape.insert(axis_reshape.begin() + axis_reshape.size() - 1, 1);
  } else if (mat2.dim() == 1) {
    axis_reshape.insert(axis_reshape.end(), 1);
  }

    at::Tensor mat2_cp;
    if (mat2.dim() == 1) {
        mat2_cp = mat2.view({1, mat2.size(0)});
    } else {
        TORCH_CHECK(mat2.dim() >= 2, "mat2.dim must be greater than or equal to 1, but got ", mat2.dim());
        mat2_cp = mat2.transpose(-2, -1);
    }
    return acl_op::npu_bmmV2(grad.view(axis_reshape), mat2_cp, sizes);
}

at::Tensor npu_bmm_v2_mat2_backward_symint(
    const at::Tensor& grad,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::SymIntArrayRef sizes_symint) {
  at::IntArrayRef sizes = c10::asIntArrayRefUnchecked(sizes_symint);
  // db = a^T * grad
  auto grad_with_full_size = grad;

  std::vector<int64_t> axis_reshape(grad.sizes().begin(), grad.sizes().end());
  if (mat1.dim() == 1) {
    axis_reshape.insert(axis_reshape.begin() + axis_reshape.size() - 1, 1);
  } else if (mat2.dim() == 1) {
    axis_reshape.insert(axis_reshape.end(), 1);
  }

  if (mat1.dim() == 1) {
    return acl_op::npu_bmmV2(mat1.view({mat1.size(0), 1}), grad.view(axis_reshape), sizes);
  }
  return acl_op::npu_bmmV2(mat1.transpose(-2, -1), grad.view(axis_reshape), sizes);
}
} // namespace acl_op
