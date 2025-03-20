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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
std::tuple<at::Tensor&, at::Tensor&> min_out_npu_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMinWithValue")
        .Input(self)
        .Output(indices)
        .Output(output)
        .Attr("dimension", dim)
        .Attr("keep_dims", keepdim)
        .Run();
    return std::tie(output, indices);
}

at::Tensor& min_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Minimum")
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
    return result;
}

at::Tensor& min_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceMin")
        .Input(self)
        .Input(dims)
        .Output(result)
        .Attr("keep_dims", keepdim)
        .Run();
    return result;
}
} // namespace

std::tuple<at::Tensor&, at::Tensor&> min_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices)
{
    c10::SmallVector<int64_t, SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

    npu_preparation::CheckOut(
        {self},
        output,
        ACL_FORMAT_ND,
        self.scalar_type(),
        output_size);

    npu_preparation::CheckOut(
        {self},
        indices,
        ACL_FORMAT_ND,
        at::ScalarType::Long,
        output_size);

    at::Tensor indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kInt);
    bool output_match = npu_utils::check_match(&output);
    bool indices_match = npu_utils::check_match(&indices);
    if (!(output_match && indices_match)) {
        at::Tensor contiguous_output = output_match ? output : npu_utils::format_contiguous(output);
        at::Tensor contiguous_indices =
            indices_match ? indices_dtype_cast : npu_utils::format_contiguous(indices_dtype_cast);
        min_out_npu_nocheck(contiguous_output, contiguous_indices, self, dim, keepdim);
        if (!output_match) {
            npu_utils::format_fresh_view(output, contiguous_output);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(indices_dtype_cast, contiguous_indices);
        }
    } else {
        min_out_npu_nocheck(output, indices_dtype_cast, self, dim, keepdim);
    }

    indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(indices_dtype_cast, at::kLong);
    indices.copy_(indices_dtype_cast);
    return std::tie(output, indices);
}

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, int64_t dim, bool keepdim)
{
    at::Tensor self_cast = self;
    if (self.dtype() == at::ScalarType::Bool) {
        self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
    }
    c10::SmallVector<int64_t, SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self_cast, dims, keepdim);

    at::Tensor outputs = npu_preparation::apply_tensor_with_format(
        output_size,
        self_cast.options(),
        ACL_FORMAT_ND);

    at::Tensor indices = npu_preparation::apply_tensor_with_format(
        output_size,
        self_cast.options().dtype(at::ScalarType::Int),
        ACL_FORMAT_NCHW);

    min_out_npu_nocheck(outputs, indices, self_cast, dim, keepdim);
    indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::ScalarType::Long);
    if (self.dtype() == at::ScalarType::Bool) {
        outputs = at_npu::native::custom_ops::npu_dtype_cast(outputs, at::ScalarType::Bool);
    }
    return std::tie(outputs, indices);
}

std::tuple<at::Tensor&, at::Tensor&> min_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices)
{
    return acl_op::min_out(self, dimname_to_position(self, dim), keepdim, output, indices);
}

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, at::Dimname dim, bool keepdim)
{
    return acl_op::min(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor& min_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    npu_preparation::CheckOut(
        {self},
        result,
        ACL_FORMAT_ND,
        self.scalar_type(),
        self.sizes());

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        min_out_npu_nocheck(contiguous_result, self, other);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        min_out_npu_nocheck(result, self, other);
    }

    return result;
}

at::Tensor minimum(const at::Tensor& self, const at::Tensor& other)
{
    auto result_type = at::result_type(self, other);
    at::Tensor self_copy = self;
    at::Tensor other_copy = other;
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        other_copy = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    } else if (at_npu::native::OpPreparation::IsCPUScalar(self)) {
        at::Scalar scalar = self.item();
        self_copy = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, self.scalar_type(), other.device());
    }
    self_copy = (self.scalar_type() != result_type) ?
        at_npu::native::custom_ops::npu_dtype_cast(self_copy, result_type) : self_copy;
    other_copy = (other.scalar_type() != result_type) ?
        at_npu::native::custom_ops::npu_dtype_cast(other_copy, result_type) : other_copy;

    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(self_copy, output_size);
    min_out_npu_nocheck(result, self_copy, other_copy);
    return result;
}

at::Tensor& minimum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    auto high_type = at::result_type(self, other);
    auto result_type = result.scalar_type();
    TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
        " can't be cast to the desired output type ", result_type,
        OPS_ERROR(ErrCode::TYPE));
    at::Tensor self_copy = self;
    at::Tensor other_copy = other;
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        other_copy = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    } else if (at_npu::native::OpPreparation::IsCPUScalar(self)) {
        at::Scalar scalar = self.item();
        self_copy = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, self.scalar_type(), other.device());
    }
    self_copy = (self.scalar_type() != result_type) ?
        at_npu::native::custom_ops::npu_dtype_cast(self_copy, result_type) : self_copy;
    other_copy = (other.scalar_type() != result_type) ?
        at_npu::native::custom_ops::npu_dtype_cast(other_copy, result_type) : other_copy;

    return acl_op::min_out(self_copy, other_copy, result);
}

at::Tensor amin(const at::Tensor& self, at::IntArrayRef dims, bool keepdim)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    int64_t npu_format = output_size.empty() ? ACL_FORMAT_NCHW : npu_preparation::get_tensor_npu_format(self);
    at::Tensor result = npu_preparation::apply_tensor_with_format(self, output_size, npu_format);
    min_out_npu_nocheck(result, self, dims, keepdim);
    return result;
}

at::Tensor min(const at::Tensor& self)
{
    c10::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
    return acl_op::amin(self, dims, false);
}

at::Tensor& amin_out(
    const at::Tensor& self,
    at::IntArrayRef dims,
    bool keepdim,
    at::Tensor& result)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    npu_preparation::CheckOut(
        {self},
        result,
        ACL_FORMAT_ND,
        self.scalar_type(),
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        min_out_npu_nocheck(contiguous_result, self, dims, keepdim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        min_out_npu_nocheck(result, self, dims, keepdim);
    }
    return result;
}
} // namespace acl_op
