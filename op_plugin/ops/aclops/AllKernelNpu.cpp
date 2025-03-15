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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor all_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dim_list,
    bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceAll")
        .Input(self)
        .Input(dim_list, at::kLong)
        .Output(result)
        .Attr("keep_dims", keepdim)
        .Run();
    return result;
}

at::Tensor& all_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    c10::SmallVector<int64_t, N> dim_list,
    bool keepdim)
{
    at::Tensor self_cast = (self.scalar_type() == at::ScalarType::Bool) ?
        self : at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Bool);
    bool result_is_bool = (result.scalar_type() == at::ScalarType::Bool);
    at::Tensor result_cast = result_is_bool ?
        result : at_npu::native::custom_ops::npu_dtype_cast(result, at::ScalarType::Bool);
    if (!npu_utils::check_match(&result_cast)) {
        at::Tensor contiguous_result_cast = npu_utils::format_contiguous(result_cast);
        all_out_npu_nocheck(contiguous_result_cast, self_cast, dim_list, keepdim);
        npu_utils::format_fresh_view(result_cast, contiguous_result_cast);
    } else {
        all_out_npu_nocheck(result_cast, self_cast, dim_list, keepdim);
    }

    if (!result_is_bool) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result.scalar_type());
        result.copy_(result_cast);
    }
    return result;
}
} // namespace

at::Tensor& all_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& out)
{
    TORCH_CHECK((out.scalar_type() == at::ScalarType::Bool || out.scalar_type() == at::ScalarType::Byte),
                "all only supports bool tensor for out, got: ", out.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    c10::SmallVector<int64_t, N> dim_list = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keepdim);
    npu_preparation::CheckOut(
        {self},
        out,
        out,
        output_size);

    return all_out_nocheck(out, self, dim_list, keepdim);
}

at::Tensor& all_out(const at::Tensor& self, at::Tensor& out)
{
    TORCH_CHECK((out.scalar_type() == at::ScalarType::Bool || out.scalar_type() == at::ScalarType::Byte),
                "all only supports bool tensor for out, got: ", out.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    at::IntArrayRef dims;
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
    npu_preparation::CheckOut(
        {self},
        out,
        out,
        output_size);
    auto dim_list = op_plugin::utils::get_dimlist_for_tensor(self);

    return all_out_nocheck(out, self, dim_list, false);
}

at::Tensor all(const at::Tensor& self, int64_t dim, bool keepdim)
{
    at::Tensor self_cast = self.scalar_type() == at::ScalarType::Bool ?
        self : at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Bool);

    if (self.dim() != 0) {
        TORCH_CHECK((dim >= -(self.dim()) && dim < self.dim()),
                    "The value of dim must be greater than or equal to -self.dim() and less than self.dim()"
                    + OPS_ERROR(ErrCode::PARAM));
    } else {
        TORCH_CHECK_INDEX((self.dim() == dim || dim == -1),
                          "Dimension out of range (expected to be in range of [-1, 0], but got ", dim, ")"
                          + OPS_ERROR(ErrCode::PARAM));
    }

    if (self.numel() == 0) {
        auto output_size = op_infer::infersize_all(self, dim);
        at::Tensor result = npu_preparation::apply_tensor(
            output_size,
            self.options().dtype(at::kBool),
            self);
        acl_op::fill_(result, 1);
        return result;
    }
    at::IntArrayRef dims(dim);
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    at::Tensor result = npu_preparation::apply_tensor(self_cast, output_size);
    all_out_npu_nocheck(result, self_cast, {dim}, keepdim);
    return result;
}

at::Tensor all(const at::Tensor& self)
{
    at::Tensor self_cast = self.scalar_type() == at::ScalarType::Bool ?
        self : at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Bool);
    if (self.numel() == 0) {
        at::Tensor result = npu_preparation::apply_tensor({}, self.options().dtype(at::kBool), self);
        acl_op::fill_(result, 1);
        return result;
    }

    at::IntArrayRef dims;
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
    at::Tensor result = npu_preparation::apply_tensor(self_cast, output_size);
    all_out_npu_nocheck(
        result,
        self_cast,
        op_plugin::utils::get_dimlist_for_tensor(self),
        false);
    return result;
}
} // namespace acl_op
