// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
const static int64_t ROTATE_HALF = 0;
const static int64_t ROTATE_INTERLEAVED = 1;

static bool isRotaryMulBackwardMixDtypeSupport(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2)
{
    return self.dtype() == r1.dtype() && self.dtype() == r2.dtype() && self.dtype() == grad.dtype() ? false : true;
}

static at::Tensor npu_dtype_cast_impl_op_api(const at::Tensor& self, at::ScalarType dtype)
{
    return self.dtype() == dtype ? self : self.to(dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2,
    int64_t mode)
{
    TORCH_CHECK(mode == ROTATE_HALF || mode == ROTATE_INTERLEAVED,
        "The mode of npu_rotary_mul_backward should be 0(rotate_half) or 1(rotate_interleaved), but got ", mode,
        OPS_ERROR(ErrCode::PARAM));
    DO_COMPATIBILITY(aclnnRotaryPositionEmbeddingGrad, acl_op::npu_rotary_mul_backward(grad, self, r1, r2, mode));
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::npu_rotary_mul_backward(grad, self, r1, r2, mode);
    }
    at::Tensor dx = npu_preparation::apply_tensor_without_format(grad.sizes(), self.options());
    at::Tensor dcos = npu_preparation::apply_tensor_without_format(r1.sizes(), self.options());
    at::Tensor dsin = npu_preparation::apply_tensor_without_format(r2.sizes(), self.options());
    bool isMixDataType = isRotaryMulBackwardMixDtypeSupport(grad, self, r1, r2);
    if (isMixDataType) {
        at::Tensor gradCast = npu_dtype_cast_impl_op_api(grad, self.scalar_type());
        at::Tensor cosCast = npu_dtype_cast_impl_op_api(r1, self.scalar_type());
        at::Tensor sinCast = npu_dtype_cast_impl_op_api(r2, self.scalar_type());
        EXEC_NPU_CMD(aclnnRotaryPositionEmbeddingGrad, gradCast, cosCast, sinCast, self, mode, dx, dcos, dsin);
    } else {
        EXEC_NPU_CMD(aclnnRotaryPositionEmbeddingGrad, grad, r1, r2, self, mode, dx, dcos, dsin);
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(dx, dcos, dsin);
}
}
