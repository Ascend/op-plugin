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
using npu_compile_type = at_npu::native::CompileType;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& argmax_out_nocheck(at::Tensor& result, const at::Tensor& input, at::Scalar& dim_scalar, bool keepdim_value)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMaxV2")
        .Input(input)
        .Input(dim_scalar, at::kInt, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Attr("keep_dims", keepdim_value)
        .Run();
    return result;
}
}

at::Tensor& argmax_out(const at::Tensor& self, at::optional<int64_t> dim, bool keepdim, at::Tensor& out)
{
    at::Tensor input = dim.has_value() ? self : self.reshape({-1});
    int64_t dim_value = dim.has_value() ? dim.value() : 0;
    bool keepdim_value = dim.has_value() ? keepdim : false;
    auto output_size = op_infer::reduce_ops_npu_output_size(input, dim_value, keepdim_value);
    npu_preparation::CheckOut(
        {self},
        out,
        npu_preparation::get_tensor_npu_format(out),
        at::kLong,
        output_size);
    at::Scalar dim_scalar = dim_value;
    at::Tensor result_cast = at_npu::native::custom_ops::npu_dtype_cast(out, at::kInt);
    argmax_out_nocheck(result_cast, input, dim_scalar, keepdim_value);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = at_npu::native::custom_ops::npu_dtype_cast(result_cast, at::kLong);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        out = at_npu::native::custom_ops::npu_dtype_cast(result_cast, at::kLong);
    }
    return out;
}

} // namespace acl_op
