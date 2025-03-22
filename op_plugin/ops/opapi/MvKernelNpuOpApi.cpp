// Copyright (c) 2022 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

at::Tensor &mv_out(const at::Tensor &self, const at::Tensor &vec, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnMv, acl_op::mv_out(self, vec, out));
    auto names = at::namedinference::propagate_names_for_addmv(self, vec, out);
    npu_preparation::check_tensor({self, vec}, out, out.scalar_type(), {self.size(0)});
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMv, self, vec, out, cube_math_type);
    at::namedinference::propagate_names_if_nonempty(out, names);
    return out;
}

at::Tensor mv(const at::Tensor &self, const at::Tensor &vec)
{
    DO_COMPATIBILITY(aclnnMv, acl_op::mv(self, vec));
    at::Tensor result;
    if (self.has_names() || vec.has_names()) {
        result = at::empty({self.size(0)}, vec.options());
    } else {
        result = npu_preparation::apply_tensor_without_format({self.size(0)}, vec.options());
    }
    auto names = at::namedinference::propagate_names_for_addmv(self, vec, result);
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnMv, self, vec, result, cube_math_type);
    at::namedinference::propagate_names_if_nonempty(result, names);
    return result;
}

}  // namespace op_api
