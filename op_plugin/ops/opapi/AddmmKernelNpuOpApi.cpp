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

namespace {
bool is_three_tensor_base_format(const at::Tensor &input0, const at::Tensor &input1, const at::Tensor &input2)
{
    return at_npu::native::FormatHelper::IsOpInputBaseFormat(input0) &&
           op_plugin::utils::is_two_tensor_base_format(input1, input2);
}

bool is_nd_nd_nz_format(const at::Tensor &self, const at::Tensor &mat1, const at::Tensor &mat2)
{
    auto dim_tensor0 = self.dim();
    // only support (n,) or (1, n) bias + 2D ND * 2D NZ
    return (dim_tensor0 == 2 || dim_tensor0 == 1) && !op_plugin::utils::is_nz_format(self) &&
           op_plugin::utils::is_nd_nz_format(mat1, mat2);
}
}  // namespace

#define DO_ADDMM_COMPATIBILITY(aclnn_nz_api, aclnn_nd_api, input0, input1, input2, aclop_func_call)          \
    do {                                                                                                     \
        if (is_three_tensor_base_format(input0, input1, input2)) {                                           \
            DO_COMPATIBILITY(aclnn_nd_api, aclop_func_call);                                                 \
        } else {                                                                                             \
            static bool is_support_soc = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&    \
                                             c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) || \
                                         (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);      \
            if (is_nd_nd_nz_format(input0, input1, input2) && is_support_soc) {                              \
                DO_COMPATIBILITY(aclnn_nz_api, aclop_func_call);                                             \
            } else {                                                                                         \
                if (!c10_npu::IsAclnnOnly()) {                                                               \
                    return aclop_func_call;                                                                  \
                }                                                                                            \
                const torch_npu::NPUStorageDesc &tensor_desc0 =                                              \
                    torch_npu::NPUBridge::GetNpuStorageImpl(input0)->npu_desc_;                              \
                const torch_npu::NPUStorageDesc &tensor_desc1 =                                              \
                    torch_npu::NPUBridge::GetNpuStorageImpl(input1)->npu_desc_;                              \
                const torch_npu::NPUStorageDesc &tensor_desc2 =                                              \
                    torch_npu::NPUBridge::GetNpuStorageImpl(input2)->npu_desc_;                              \
                TORCH_CHECK(false,                                                                           \
                    "matmul got not support format in current device: ",                                     \
                    "(",                                                                                     \
                    tensor_desc0.npu_format_,                                                                \
                    ", ",                                                                                    \
                    tensor_desc1.npu_format_,                                                                \
                    ", ",                                                                                    \
                    tensor_desc2.npu_format_,                                                                \
                    ")",                                                                                     \
                    OPS_ERROR(ErrCode::PARAM));                                                              \
            }                                                                                                \
        }                                                                                                    \
    } while (0)

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor &addmm_out(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha,
    at::Tensor &out)
{
    DO_ADDMM_COMPATIBILITY(aclnnAddmmWeightNz, aclnnAddmm, self, mat1, mat2,
                           acl_op::addmm_out(self, mat1, mat2, beta, alpha, out));
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    auto output_size = op_infer::addmm_npu_output_size(self, mat1, mat2);
    npu_preparation::check_tensor({self, mat1, mat2}, out, out.scalar_type(), output_size);
    if (is_nd_nd_nz_format(self, mat1, mat2)) {
        EXEC_NPU_CMD(aclnnAddmmWeightNz, self, mat1, mat2, beta, alpha, out, cube_math_type);
    } else {
        EXEC_NPU_CMD(aclnnAddmm, self, mat1, mat2, beta, alpha, out, cube_math_type);
    }
    auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self);
    at::namedinference::propagate_names_if_nonempty(out, names);

    return out;
}

at::Tensor addmm(
    const at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
    DO_ADDMM_COMPATIBILITY(aclnnAddmmWeightNz, aclnnAddmm, self, mat1, mat2,
                           acl_op::addmm(self, mat1, mat2, beta, alpha));
    auto output_size = op_infer::addmm_npu_output_size(self, mat1, mat2);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    if (is_nd_nd_nz_format(self, mat1, mat2)) {
        EXEC_NPU_CMD(aclnnAddmmWeightNz, self, mat1, mat2, beta, alpha, result, cube_math_type);
    } else {
        EXEC_NPU_CMD(aclnnAddmm, self, mat1, mat2, beta, alpha, result, cube_math_type);
    }
    auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self);
    at::namedinference::propagate_names_if_nonempty(result, names);
    FLOP_COUNT(FlopCounter::addmm_flop, mat1, mat2);
    return result;
}

at::Tensor &addmm_(
    at::Tensor &self,
    const at::Tensor &mat1,
    const at::Tensor &mat2,
    const at::Scalar &beta,
    const at::Scalar &alpha)
{
    DO_COMPATIBILITY(aclnnInplaceAddmm, acl_op::addmm_(self, mat1, mat2, beta, alpha));
    auto output_size = op_infer::addmm_npu_output_size(self, mat1, mat2);
    npu_preparation::check_tensor({self, mat1, mat2}, self, self.scalar_type(), output_size);

    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnInplaceAddmm, self, mat1, mat2, beta, alpha, cube_math_type);

    auto names = at::namedinference::propagate_names_for_addmm(mat1, mat2, self);
    at::namedinference::propagate_names_if_nonempty(self, names);

    return self;
}

}
