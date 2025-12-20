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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor mm(const at::Tensor &self, const at::Tensor &mat2)
{
    TORCH_CHECK(self.dim() == 2, "self must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x",  self.sizes()[1], " and ",  mat2.sizes()[0], "x", mat2.sizes()[1], ")"
    );

    auto names = at::namedinference::compute_matmul_outnames(self, mat2);
    DO_MATMUL_COMPATIBILITY(aclnnMatmulWeightNz, aclnnMm, self, mat2, acl_op::mm(self, mat2));
    // aclnn
    auto output_size = {self.size(0), mat2.size(1)};
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    if (op_plugin::utils::is_nd_nz_format(self, mat2)) {
        EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, result, cube_math_type);
    } else {
        int8_t cube_math_type_passthrough = npu_preparation::get_cube_math_type();
        if (cube_math_type_passthrough >= 0) {
            cube_math_type = cube_math_type_passthrough;
        }
        EXEC_NPU_CMD(aclnnMm, self, mat2, result, cube_math_type);
    }
    at::namedinference::propagate_names_if_nonempty(result, names);
    FLOP_COUNT(FlopCounter::mm_flop, self, mat2);
    return result;
}

at::Tensor &mm_out(const at::Tensor &self, const at::Tensor &mat2, at::Tensor &out)
{
    TORCH_CHECK(self.dim() == 2, "self must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(
      self.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      self.sizes()[0], "x",  self.sizes()[1], " and ",  mat2.sizes()[0], "x", mat2.sizes()[1], ")"
    );

    auto names = at::namedinference::compute_matmul_outnames(self, mat2);
    DO_MATMUL_COMPATIBILITY(aclnnMatmulWeightNz, aclnnMm, self, mat2, acl_op::mm_out(self, mat2, out));
    // aclnn
    auto output_size = {self.size(0), mat2.size(1)};
    npu_preparation::check_tensor({self, mat2}, out, self.scalar_type(), output_size);
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    if (op_plugin::utils::is_nd_nz_format(self, mat2)) {
        EXEC_NPU_CMD(aclnnMatmulWeightNz, self, mat2, out, cube_math_type);
    } else {
        int8_t cube_math_type_passthrough = npu_preparation::get_cube_math_type();
        if (cube_math_type_passthrough >= 0) {
            cube_math_type = cube_math_type_passthrough;
        }
        EXEC_NPU_CMD(aclnnMm, self, mat2, out, cube_math_type);
    }
    at::namedinference::propagate_names_if_nonempty(out, names);
    return out;
}

}  // namespace op_api
