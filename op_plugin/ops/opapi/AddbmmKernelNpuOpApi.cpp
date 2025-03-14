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

at::Tensor &addbmm_out(const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2,
                       const at::Scalar &beta, const at::Scalar &alpha, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnAddbmm, acl_op::addbmm_out(self, batch1, batch2, beta, alpha, out));
    TORCH_CHECK(batch1.ndimension() >= 3, "Expected least 3D tensor, but got a tensor with sizes ", batch1.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(batch2.ndimension() >= 3, "Expected least 3D tensor, but got a tensor with sizes ", batch2.dim(),
        OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::addbmm_npu_output_size(self, batch1, batch2);
    npu_preparation::check_tensor({self, batch1, batch2}, out, self.scalar_type(), output_size);
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnAddbmm, self, batch1, batch2, beta, alpha, out, cube_math_type);

    return out;
}

at::Tensor addbmm(const at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, const at::Scalar &beta,
                  const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnAddbmm, acl_op::addbmm(self, batch1, batch2, beta, alpha));
    TORCH_CHECK(batch1.ndimension() >= 3, "Expected least 3D tensor, but got a tensor with sizes ", batch1.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(batch2.ndimension() >= 3, "Expected least 3D tensor, but got a tensor with sizes ", batch2.dim(),
        OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::addbmm_npu_output_size(self, batch1, batch2);
    at::Tensor result = npu_preparation::apply_tensor_with_sizes(output_size, self.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnAddbmm, self, batch1, batch2, beta, alpha, result, cube_math_type);

    return result;
}

at::Tensor &addbmm_(at::Tensor &self, const at::Tensor &batch1, const at::Tensor &batch2, const at::Scalar &beta,
                    const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnAddbmm, acl_op::addbmm_(self, batch1, batch2, beta, alpha));

    op_api::addbmm_out(self, batch1, batch2, beta, alpha, self);
    return self;
}

}
