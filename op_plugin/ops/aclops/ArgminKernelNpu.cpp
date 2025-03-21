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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& argmin_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& dim,
    bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMin")
        .Input(self)
        .Input(dim, at::kInt)
        .Output(result)
        .Attr("keep_dims", keepdim)
        .Run();
    return result;
}
} // namespace

at::Tensor argmin(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim)
{
    TORCH_CHECK(
        self.numel() > 0,
        "cannot perform reduction function argmin on a "
        "tensor with no elements because the operation does not have an identity"
        + OPS_ERROR(ErrCode::PARAM));
    at::Tensor input = dim.has_value() ? self : self.reshape({-1});
    int64_t dim_value = dim.has_value() ? dim.value() : 0;
    bool keepdim_value = dim.has_value() ? keepdim : false;
    auto output_size = op_infer::reduce_ops_npu_output_size(input, dim_value, keepdim_value);
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size,
        self.options().dtype(at::kInt),
        ACL_FORMAT_ND);
    c10::Scalar dim_scalar = dim_value;

    argmin_out_nocheck(result, input, dim_scalar, keepdim_value);
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kLong);
    return result;
}

at::Tensor& argmin_out(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& result)
{
    TORCH_CHECK(
        self.numel() > 0,
        "cannot perform reduction function argmin on a "
        "tensor with no elements because the operation does not have an identity"
        + OPS_ERROR(ErrCode::PARAM));
    at::Tensor input = dim.has_value() ? self : self.reshape({-1});
    int64_t dim_value = dim.has_value() ? dim.value() : 0;
    bool keepdim_value = dim.has_value() ? keepdim : false;
    auto output_size = op_infer::reduce_ops_npu_output_size(input, dim_value, keepdim_value);

    npu_preparation::CheckOut(
        {self},
        result,
        npu_preparation::get_tensor_npu_format(result),
        at::kLong,
        output_size);

    c10::Scalar dim_scalar = dim_value;
    at::Tensor result_cast = at_npu::native::custom_ops::npu_dtype_cast(result, at::kInt);
    argmin_out_nocheck(result_cast, input, dim_scalar, keepdim_value);
    result = at_npu::native::custom_ops::npu_dtype_cast(result_cast, at::kLong);
    return result;
}
} // namespace acl_op
