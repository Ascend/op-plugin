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

static bool isRotaryMulMixDtypeSupport(
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2)
{
    return self.dtype() == r1.dtype() && self.dtype() == r2.dtype() ? false : true;
}

static at::Tensor npu_dtype_cast_impl_op_api(const at::Tensor& self, at::ScalarType dtype)
{
    return self.dtype() == dtype ? self : self.to(dtype);
}

at::Tensor npu_rotary_mul(
    const at::Tensor& self,
    const at::Tensor& r1,
    const at::Tensor& r2,
    c10::string_view rotary_mode)
{
    TORCH_CHECK(rotary_mode == "half" || rotary_mode == "interleave",
        "The rotary_mode of npu_rotary_mul should be half or interleave, but got ", rotary_mode,
        OPS_ERROR(ErrCode::PARAM));
    DO_COMPATIBILITY(aclnnRotaryPositionEmbedding, acl_op::npu_rotary_mul(self, r1, r2, rotary_mode));
    int64_t mode = op_plugin::utils::get_rotary_mode(rotary_mode);
    if (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910B1) {
        return acl_op::npu_rotary_mul(self, r1, r2, rotary_mode);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(), self.options());
    bool isMixDataType = isRotaryMulMixDtypeSupport(self, r1, r2);
    if (isMixDataType) {
        at::Tensor cosCast = npu_dtype_cast_impl_op_api(r1, self.scalar_type());
        at::Tensor sinCast = npu_dtype_cast_impl_op_api(r2, self.scalar_type());
        EXEC_NPU_CMD(aclnnRotaryPositionEmbedding, self, cosCast, sinCast, mode, result);
    } else {
        EXEC_NPU_CMD(aclnnRotaryPositionEmbedding, self, r1, r2, mode, result);
    }
    return result;
}
}
