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

namespace {
    at::Tensor &baddbmm_nocheck(
        at::Tensor &result,
        const at::Tensor &self,
        const at::Tensor &tensor1,
        const at::Tensor &tensor2,
        at::Scalar beta,
        at::Scalar alpha) {
        auto output_size = op_infer::baddbmm_npu_output_size(tensor1, tensor2);
        at::Tensor batch_matmul_tensor = npu_preparation::apply_tensor(self, output_size);
        bool is_self_t = op_plugin::utils::is_transpose_last_two_dims(tensor1);
        bool is_mat2_t = op_plugin::utils::is_transpose_last_two_dims(tensor2);
        at::Tensor contiguous_self = is_self_t ? tensor1 : npu_utils::format_contiguous(tensor1);
        at::Tensor contiguous_mat2 = is_mat2_t ? tensor2 : npu_utils::format_contiguous(tensor2);

        at_npu::native::OpCommand cmd;
        cmd.Name("BatchMatMul")
            .InputWithoutContiguous(contiguous_self)
            .InputWithoutContiguous(contiguous_mat2)
            .Output(batch_matmul_tensor)
            .Attr("adj_x1", is_self_t)
            .Attr("adj_x2", is_mat2_t)
            .Run();

        at::Tensor alpha_mul_tensor = acl_op::mul(batch_matmul_tensor, alpha);
        at::Tensor beta_mul_tensor = (beta.toFloat() == 0.0) ? at::zeros_like(self) : acl_op::mul(self, beta);
        acl_op::add_out(alpha_mul_tensor, beta_mul_tensor, 1, result);
        return result;
    }
} // namespace

at::Tensor &baddbmm_out(
    const at::Tensor &self,
    const at::Tensor &tensor1,
    const at::Tensor &tensor2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &result) {
    npu_preparation::CheckOut(
        {self, tensor1, tensor2},
        result,
        self);
    if (!result.is_contiguous()) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        baddbmm_nocheck(contiguous_result, self, tensor1, tensor2, beta, alpha);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        baddbmm_nocheck(result, self, tensor1, tensor2, beta, alpha);
    }
    return result;
}

at::Tensor baddbmm(
    const at::Tensor &self,
    const at::Tensor &tensor1,
    const at::Tensor &tensor2,
    const at::Scalar &beta,
    const at::Scalar &alpha) {
    at::Tensor result = npu_preparation::apply_tensor(self);
    baddbmm_nocheck(result, self, tensor1, tensor2, beta, alpha);
    return result;
}

at::Tensor &baddbmm_(
    at::Tensor &self,
    const at::Tensor &tensor1,
    const at::Tensor &tensor2,
    const at::Scalar &beta,
    const at::Scalar &alpha) {
    return acl_op::baddbmm_out(self, tensor1, tensor2, beta, alpha, self);
}
} // namespace acl_op
