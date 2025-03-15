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

#include <ATen/WrapDimUtilsMulti.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& sum_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim)
{
    at::dim_list_to_bitset(dim, self.dim());
    c10::SmallVector<int64_t, N> dim_list = dim.empty() ? op_plugin::utils::get_dimlist_for_tensor(self) :
        c10::SmallVector<int64_t, N>(dim);
    at_npu::native::OpCommand cmd;
    cmd.Name("ReduceSum")
        .Input(self)
        .Input(dim_list, at::kLong)
        .Output(result)
        .Attr("keep_dims", keepdim)
        .Run();
    return result;
}

at::Tensor check_dtype(
    const at::Tensor &self,
    c10::ScalarType out_type)
{
    if (isIntegralType(out_type, true)) {
        out_type = at::kFloat;
    }
    at::Tensor self_cp = (self.scalar_type() == out_type) ? self :
        acl_op::npu_dtype_cast(self, out_type);
    return self_cp;
}
} // namespace

at::Tensor& sum_out_common_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    auto output_size = op_infer::sum_npu_output_size(self, dim, keepdim);
    auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();

    npu_preparation::CheckOut(
        {self},
        result,
        ACL_FORMAT_ND,
        res_type,
        output_size);

    if (self.numel() == 0) {
        at::Tensor result_cast = at::empty(output_size, self.options().dtype(res_type));
        result.copy_(result_cast);
        return result;
    }

    at::Tensor self_cp = check_dtype(self, res_type);
    at::Tensor result_cp = result.scalar_type() == self_cp.scalar_type() ? result :
        acl_op::npu_dtype_cast(result, self_cp.scalar_type());
    if (!npu_utils::check_match(&result_cp)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result_cp);
        sum_out_npu_nocheck(contiguous_result, self_cp, dim, keepdim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        sum_out_npu_nocheck(result_cp, self_cp, dim, keepdim);
    }

    if (result_cp.scalar_type() != res_type) {
        result_cp = acl_op::npu_dtype_cast(result_cp, res_type);
        result.copy_(result_cp);
    } else {
        result = result_cp;
    }
    return result;
}


at::Tensor sum_common_nocheck(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype)
{
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto out_type = self.scalar_type();

    if (dtype.has_value()) {
        out_type = dtype.value();
    } else if (isIntegralType(out_type, true)) {
        out_type = at::kLong;
    }

    if (self.numel() == 0) {
        return at::zeros(output_size, self.options().dtype(out_type));
    }

    at::Tensor self_cp = check_dtype(self, out_type);
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size, self_cp.options(), ACL_FORMAT_ND);
    sum_out_npu_nocheck(result, self_cp, dim, keepdim);

    if (result.scalar_type() != out_type) {
        result = acl_op::npu_dtype_cast(result, out_type);
    }
    return result;
}
} // namespace acl_op
