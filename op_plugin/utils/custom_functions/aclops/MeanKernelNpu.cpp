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
using npu_utils = at_npu::native::NpuUtils;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& mean_out_no_dtype_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    if (self.numel() == 0 && dim.size() == 0) {
        // In this scenario, needs to return nan. And the nan of the NPU can only be fp32.
        result = acl_op::npu_dtype_cast(result, at::kFloat).fill_(NAN);
        return result;
    }

    c10::SmallVector<int64_t, N> dim_vec;
    if (dim.empty()) {
        dim_vec = op_plugin::utils::get_dimlist_for_tensor(self);
    } else {
        dim_vec = op_infer::array_to_small_vector(dim);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceMean")
        .Input(self)
        .Input(dim_vec, at::kLong)
        .Output(result)
        .Attr("keep_dims", keepdim)
        .Run();
    return result;
}

at::Tensor& mean_out_no_dtype(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    int64_t npu_format = npu_preparation::get_tensor_npu_format(result);
    // scalar scene and rank=1 scene do not support NZ
    if (output_size.size() < 2) {
        npu_format = ACL_FORMAT_NCHW;
    }
    npu_preparation::CheckOut(
        {self},
        result,
        npu_format,
        self.scalar_type(),
        output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        mean_out_no_dtype_nocheck(contiguous_result, self, dim, keepdim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        mean_out_no_dtype_nocheck(result, self, dim, keepdim);
    }
    return result;
}
} // namespace

at::Tensor& mean_out_common_nocheck(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype,
    at::Tensor& result)
{
    c10::ScalarType dst_type;
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (result.defined()) {
        dst_type = result.scalar_type();
    } else {
        dst_type = self.scalar_type();
    }

    if (dst_type == self.scalar_type()) {
        mean_out_no_dtype(result, self, dim, keepdim);
        return result;
    }

    mean_out_no_dtype(result, self.toType(dst_type), dim, keepdim);
    return result;
}

at::Tensor mean_common_nocheck(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    c10::ScalarType dst_type = dtype.has_value() ? dtype.value() : self.scalar_type();

    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);

    int64_t npu_format = npu_preparation::get_tensor_npu_format(self);
    if (output_size.empty()) {
        npu_format = ACL_FORMAT_NCHW;
    }

    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, self.options().dtype(dst_type), npu_format);

    mean_out(self, dim, keepdim, dtype, result);
    return result;
}
} // namespace acl_op
