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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& addmv_out(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& result) {
    npu_utils::check_1d(vec, "vec", "addmv");

    at::Tensor mat1 = vec.unsqueeze(1);
    at::Tensor mat_alpha = at::mul(mat, alpha);
    at::Tensor mm_mul_result = at::mm(mat_alpha, mat1);
    at::Tensor mm_mul_result1 = mm_mul_result.squeeze();

    auto output_size = op_infer::addmv_npu_output_size(self, mat);
    if (!result.sizes().equals(output_size)) {
        result.resize_(output_size);
    }
    // matmul*alpha+self*beta
    at::add_out(result, mm_mul_result1, self, beta);
    return result;
}

at::Tensor addmv(
    const at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
    auto output_size = op_infer::addmv_npu_output_size(self, mat);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    addmv_out(self, mat, vec, beta, alpha, result);
    return result;
}

at::Tensor& addmv_(
    at::Tensor& self,
    const at::Tensor& mat,
    const at::Tensor& vec,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
    npu_preparation::CheckMemory({self, mat, vec}, {self});
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        acl_op::addmv_out(contiguous_self, mat, vec, beta, alpha, contiguous_self);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        acl_op::addmv_out(self, mat, vec, beta, alpha, self);
    }
    return self;
}
} // namespace acl_op
