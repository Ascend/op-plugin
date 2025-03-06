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
#include <ATen/native/TypeProperties.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static at::Tensor &addmv_out_op_api(const at::Tensor &self, const at::Tensor &mat, const at::Tensor &vec,
                                    const at::Scalar &beta, const at::Scalar &alpha, at::Tensor &result)
{
    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    EXEC_NPU_CMD(aclnnAddmv, self, mat, vec, alpha, beta, result, cube_math_type);
    return result;
}

at::Tensor &addmv_out(const at::Tensor &self, const at::Tensor &mat, const at::Tensor &vec, const at::Scalar &beta,
                      const at::Scalar &alpha, at::Tensor &result)
{
    auto names = at::namedinference::propagate_names_for_addmv(mat, vec, self);
    DO_COMPATIBILITY(aclnnAddmv, acl_op::addmv_out(self, mat, vec, beta, alpha, result));
    auto output_size = op_infer::addmv_npu_output_size(self, mat);
    if (!result.sizes().equals(output_size)) {
        result.resize_(output_size);
    }
    addmv_out_op_api(self, mat, vec, beta, alpha, result);
    at::namedinference::propagate_names_if_nonempty(result, names);
    return result;
}

at::Tensor addmv(const at::Tensor &self, const at::Tensor &mat, const at::Tensor &vec, const at::Scalar &beta,
                 const at::Scalar &alpha)
{
    auto names = at::namedinference::propagate_names_for_addmv(mat, vec, self);
    DO_COMPATIBILITY(aclnnAddmv, acl_op::addmv(self, mat, vec, beta, alpha));
    auto output_size = op_infer::addmv_npu_output_size(self, mat);
    at::ScalarType promote_dtype = at::native::result_type({self, mat, vec});
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(promote_dtype));
    addmv_out_op_api(self, mat, vec, beta, alpha, result);
    at::namedinference::propagate_names_if_nonempty(result, names);
    return result;
}

at::Tensor &addmv_(at::Tensor &self, const at::Tensor &mat, const at::Tensor &vec, const at::Scalar &beta,
                   const at::Scalar &alpha)
{
    auto names = at::namedinference::propagate_names_for_addmv(mat, vec, self);
    DO_COMPATIBILITY(aclnnAddmv, acl_op::addmv_(self, mat, vec, beta, alpha));
    npu_preparation::check_memory({self, mat, vec}, {self});
    addmv_out_op_api(self, mat, vec, beta, alpha, self);
    at::namedinference::propagate_names_if_nonempty(self, names);
    return self;
}
} // namespace op_api
