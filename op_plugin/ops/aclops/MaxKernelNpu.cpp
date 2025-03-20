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
std::tuple<at::Tensor &, at::Tensor &> max_out_npu_nocheck(at::Tensor &output, at::Tensor &indices,
                                                           const at::Tensor &self, int64_t dim, bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMaxWithValue")
        .Input(self)
        .Output(indices)
        .Output(output)
        .Attr("dimension", dim)
        .Attr("keep_dims", keepdim)
        .Run();
    return std::tie(output, indices);
}

at::Tensor &max_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Maximum").Input(self).Input(other).Output(result).Run();
    return result;
}

at::Tensor &max_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, at::IntArrayRef dims, bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceMax").Input(self).Input(dims).Output(result).Attr("keep_dims", keepdim).Run();
    return result;
}

at::Tensor &max_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Maximum").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}
} // namespace

std::tuple<at::Tensor &, at::Tensor &> max_out(const at::Tensor &self, int64_t dim, bool keepdim, at::Tensor &max,
                                               at::Tensor &max_values)
{
    at::SmallVector<int64_t, SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

    npu_preparation::CheckOut({self}, max, ACL_FORMAT_ND, self.scalar_type(), output_size);

    npu_preparation::CheckOut({self}, max_values, ACL_FORMAT_ND, at::ScalarType::Long, output_size);

    at::Tensor indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(max_values, at::ScalarType::Int);
    bool output_match = npu_utils::check_match(&max);
    bool indices_match = npu_utils::check_match(&indices_dtype_cast);
    if (!(output_match && indices_match)) {
        at::Tensor contiguous_output = output_match ? max : npu_utils::format_contiguous(max);
        at::Tensor contiguous_indices =
            indices_match ? indices_dtype_cast : npu_utils::format_contiguous(indices_dtype_cast);

        max_out_npu_nocheck(contiguous_output, contiguous_indices, self, dim, keepdim);

        if (!output_match) {
            npu_utils::format_fresh_view(max, contiguous_output);
        }
        if (!indices_match) {
            npu_utils::format_fresh_view(indices_dtype_cast, contiguous_indices);
        }
    } else {
        max_out_npu_nocheck(max, indices_dtype_cast, self, dim, keepdim);
    }

    indices_dtype_cast = at_npu::native::custom_ops::npu_dtype_cast(indices_dtype_cast, at::ScalarType::Long);
    max_values.copy_(indices_dtype_cast);
    return std::tie(max, max_values);
}

std::tuple<at::Tensor &, at::Tensor &> max_out(const at::Tensor &self, at::Dimname dim, bool keepdim,
                                               at::Tensor &max, at::Tensor &max_values)
{
    return acl_op::max_out(self, dimname_to_position(self, dim), keepdim, max, max_values);
}

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor &self, int64_t dim, bool keepdim)
{
    at::Tensor self_cast = self;
    if (self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int) {
        self_cast = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
    }

    at::SmallVector<int64_t, SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self_cast, dims, keepdim);

    at::Tensor outputs = npu_preparation::apply_tensor_with_format(output_size, self_cast.options(), ACL_FORMAT_ND);
    at::Tensor indices = npu_preparation::apply_tensor_with_format(
        output_size, self_cast.options().dtype(at::ScalarType::Int), ACL_FORMAT_ND);

    max_out_npu_nocheck(outputs, indices, self_cast, dim, keepdim);
    indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::ScalarType::Long);

    if (self.dtype() == at::ScalarType::Bool || self.dtype() == at::ScalarType::Int) {
        outputs = at_npu::native::custom_ops::npu_dtype_cast(outputs, self.scalar_type());
    }

    return std::tie(outputs, indices);
}

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor &self, at::Dimname dim, bool keepdim)
{
    return at::max(self, dimname_to_position(self, dim), keepdim);
}

at::Tensor &max_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);

    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor self_copy = (self.scalar_type() != high_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                               at_npu::native::custom_ops::npu_dtype_cast(self, high_type) :
                               self;
    at::Tensor other_copy = (other.scalar_type() != high_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                                at_npu::native::custom_ops::npu_dtype_cast(other, high_type) :
                                other;

    npu_preparation::CheckOut({self_copy, other_copy}, out, self_copy, output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        max_out_npu_nocheck(contiguous_result, self_copy, other_copy);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        max_out_npu_nocheck(out, self_copy, other_copy);
    }

    return out;
}

at::Tensor &maximum_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self, other}, out, self, output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        max_out_npu_nocheck(contiguous_result, self, other);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        max_out_npu_nocheck(out, self, other);
    }

    return out;
}

at::Tensor maximum(const at::Tensor &self, const at::Tensor &other)
{
    auto output_size_diff = self.sizes();
    at::Tensor result_diff = npu_preparation::apply_tensor(self, output_size_diff);
    if (npu_preparation::IsCPUScalar(other)) {
        max_out_npu_nocheck(result_diff, self, other.item());
        return result_diff;
    }
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::ScalarType high_type = at::native::result_type(self, other);
    at::Tensor self_copy = (self.scalar_type() != high_type && !npu_preparation::is_scalar_wrapped_to_tensor(self)) ?
                               at_npu::native::custom_ops::npu_dtype_cast(self, high_type) :
                               self;
    at::Tensor other_copy = (other.scalar_type() != high_type && !npu_preparation::is_scalar_wrapped_to_tensor(other)) ?
                                at_npu::native::custom_ops::npu_dtype_cast(other, high_type) :
                                other;
    at::Tensor result = npu_preparation::apply_tensor(self_copy, output_size);
    max_out_npu_nocheck(result, self_copy, other_copy);
    return result;
}

at::Tensor amax(const at::Tensor &self, at::IntArrayRef dim, bool keepdim)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    int64_t npu_format = npu_preparation::get_tensor_npu_format(self);
    if (output_size.empty()) {
        npu_format = ACL_FORMAT_ND;
    }
    at::Tensor result = npu_preparation::apply_tensor_with_format(self, output_size, npu_format);
    max_out_npu_nocheck(result, self, dim, keepdim);
    return result;
}

at::Tensor max(const at::Tensor &self)
{
    at::SmallVector<int64_t, SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
    return acl_op::amax(self, dims, false);
}

at::Tensor &amax_out(const at::Tensor &self, at::IntArrayRef dim, bool keepdim, at::Tensor &out)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    npu_preparation::CheckOut({self}, out, ACL_FORMAT_ND, self.scalar_type(), output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        max_out_npu_nocheck(contiguous_result, self, dim, keepdim);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        max_out_npu_nocheck(out, self, dim, keepdim);
    }

    return out;
}
} // namespace acl_op
