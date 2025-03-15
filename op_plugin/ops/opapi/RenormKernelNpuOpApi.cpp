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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& renorm_out(const at::Tensor& self,
                       const at::Scalar& p,
                       int64_t dim,
                       const at::Scalar& maxnorm,
                       at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnRenorm, acl_op::renorm_out(self, p, dim, maxnorm, out));

    auto dim_post_expr = self.dim();
    if (dim_post_expr <= 0) {
        dim_post_expr = 1;
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    auto output_size = op_infer::input_same_output_size(self);
    npu_preparation::check_tensor(
        {self},
        out,
        out.scalar_type(),
        output_size);

    EXEC_NPU_CMD(aclnnRenorm, self, p, dim, maxnorm, out);
    return out;
}

at::Tensor renorm(const at::Tensor& self,
                  const at::Scalar& p,
                  int64_t dim,
                  const at::Scalar& maxnorm)
{
    DO_COMPATIBILITY(aclnnRenorm, acl_op::renorm(self, p, dim, maxnorm));

    auto dim_post_expr = self.dim();
    if (dim_post_expr <= 0) {
        dim_post_expr = 1;
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    // calculate the output size
    auto output_size = op_infer::input_same_output_size(self);

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnRenorm, self, p, dim, maxnorm, result);
    return result;
}

at::Tensor& renorm_(at::Tensor& self, const at::Scalar& p, int64_t dim, const at::Scalar& maxnorm)
{
    DO_COMPATIBILITY(aclnnInplaceRenorm, acl_op::renorm_(self, p, dim, maxnorm));

    auto dim_post_expr = self.dim();
    if (dim_post_expr <= 0) {
        dim_post_expr = 1;
    }
    if (dim < 0) {
        dim += dim_post_expr;
    }
    EXEC_NPU_CMD(aclnnInplaceRenorm, self, p, dim, maxnorm);
    return self;
}

}
