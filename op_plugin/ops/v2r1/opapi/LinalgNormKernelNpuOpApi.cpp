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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor linalg_vector_norm(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype)
{
    DO_COMPATIBILITY(aclnnLinalgVectorNorm,
                     acl_op::linalg_vector_norm(self, scalar_ord, opt_dim, keepdim, opt_dtype));
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(dtype));
    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self, scalar_ord, dim, keepdim, dtype, out);
    return out;
}

at::Tensor& linalg_vector_norm_out(
    const at::Tensor& self,
    const at::Scalar& scalar_ord,
    at::OptionalIntArrayRef opt_dim,
    bool keepdim,
    at::optional<at::ScalarType> opt_dtype,
    at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnLinalgVectorNorm,
                     acl_op::linalg_vector_norm_out(self, scalar_ord, opt_dim, keepdim, opt_dtype, result));
    auto dim = opt_dim.value_or(at::IntArrayRef{});
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto dtype = opt_dtype.has_value() ? opt_dtype.value() : self.scalar_type();
    npu_preparation::check_tensor(
        {self},
        result,
        dtype,
        output_size);

    EXEC_NPU_CMD(aclnnLinalgVectorNorm, self, scalar_ord, dim, keepdim, dtype, result);
    return result;
}
} // namespace op_api
