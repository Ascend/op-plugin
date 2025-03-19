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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;


at::Tensor &addbmm_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &batch1,
                                   const at::Tensor &batch2)
{
    bool is_batch1_t = op_plugin::utils::is_transpose_last_two_dims(batch1);
    bool is_batch2_t = op_plugin::utils::is_transpose_last_two_dims(batch2);
    at::Tensor contiguous_batch1 = is_batch1_t ? batch1 : npu_utils::format_contiguous_add_copy_optimize(batch1);
    at::Tensor contiguous_batch2 = is_batch2_t ? batch2 : npu_utils::format_contiguous_add_copy_optimize(batch2);

    at_npu::native::OpCommand cmd;
    cmd.Name("BatchMatMul")
        .InputWithoutContiguous(contiguous_batch1)
        .InputWithoutContiguous(contiguous_batch2)
        .Input(self)
        .Output(result)
        .Attr("adj_x1", is_batch1_t)
        .Attr("adj_x2", is_batch2_t)
        .Run();
    return result;
}

at::Tensor& addbmm_out(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out)
{
    TORCH_CHECK(batch1.dim() >= 2 && batch2.dim() >= 3,
        "batch1 is expected to be at least 2D and batch2 is expected to be at least 3D, but got batch1: ",
        batch1.dim(), "D, batch2: ", batch2.dim(), "D" + OPS_ERROR(ErrCode::PARAM));
    static const bool is_support_nd_out = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1;
    bool check_bias_shape = ((self.dim() == 1 || (self.dim() == 2 && self.size(0) == 1)) && batch1.size(0) == 1);
    std::vector<int64_t> dims = {batch1.size(1), batch2.size(2)};
    if (check_bias_shape && is_support_nd_out) {
        auto output_size = {batch1.size(0), batch1.size(1), batch2.size(2)};
        at::Tensor biasbmm_result = npu_preparation::apply_tensor(self, output_size);
        if ((std::abs(beta.toFloat() - 1.0f) <= std::numeric_limits<float>::epsilon()) &&
            (std::abs(alpha.toFloat() - 1.0f) <= std::numeric_limits<float>::epsilon())) {
            acl_op::addbmm_out_npu_nocheck(biasbmm_result, self, batch1, batch2);
        } else {
            at::Tensor mul_result = at::mul(batch1, alpha);
            at::Tensor bias = at::mul(self, beta);
            acl_op::addbmm_out_npu_nocheck(biasbmm_result, bias, mul_result, batch2);
        }
        out = at::sum_to(biasbmm_result, dims);
    } else {
        at::Tensor mul_result = at::mul(batch1, alpha);
        at::Tensor bmm_result = at::bmm(mul_result, batch2);
        at::Tensor sum_result = at::sum_to(bmm_result, dims);
        // sum_result + self*beta
        at::add_out(out, sum_result, self, beta);
    }
    return out;
}

at::Tensor addbmm(
    const at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha)
{
    auto output_size = op_infer::addbmm_npu_output_size(self, batch1, batch2);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    acl_op::addbmm_out(self, batch1, batch2, beta, alpha, result);
    return result;
}

at::Tensor& addbmm_(
    at::Tensor& self,
    const at::Tensor& batch1,
    const at::Tensor& batch2,
    const at::Scalar& beta,
    const at::Scalar& alpha)
{
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
