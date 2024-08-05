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
at::Tensor &scatter_out_npu_nocheck(at::Tensor &result, at::Tensor &self, int64_t dim, const at::Tensor &index,
                                    const at::Tensor &src)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterElements")
        .Input(self)
        .Input(index)
        .Input(src)
        .Output(result)
        .Attr("axis", dim)
        .Run();
    return result;
}

at::Tensor &scatter_out_nocheck(at::Tensor &result, const at::Tensor &self_ex, int64_t dim, const at::Tensor &index,
                                const at::Tensor &src_ex)
{
    at::Tensor self = self_ex;
    at::Tensor result_ex = result;
    at::ScalarType self_type = self.scalar_type();
    if (self_type == at::ScalarType::Half) {
        self = at_npu::native::custom_ops::npu_dtype_cast(self, at::ScalarType::Float);
        result_ex = at_npu::native::custom_ops::npu_dtype_cast(result_ex, at::ScalarType::Float);
    }

    at::Tensor src(src_ex);
    if (src.scalar_type() != self.scalar_type()) {
        src = at_npu::native::custom_ops::npu_dtype_cast(src, self.scalar_type());
    }
    scatter_out_npu_nocheck(result_ex, self, dim, index, src);

    if (result_ex.scalar_type() != self_type) {
        result_ex = at_npu::native::custom_ops::npu_dtype_cast(result_ex, self_type);
        result.copy_(result_ex);
    } else {
        result = result_ex;
    }

    return result;
}

at::Tensor &scatter_inplace_nocheck(at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src_ex)
{
    at::ScalarType self_type = self.scalar_type();
    at::Tensor self_ex(self);
    if (self_type == at::ScalarType::Half) {
        self_ex = acl_op::npu_dtype_cast(self, at::ScalarType::Float);
    }

    at::Tensor src(src_ex);
    if (src.scalar_type() != self_ex.scalar_type()) {
        src = acl_op::npu_dtype_cast(src, self_ex.scalar_type());
    }

    scatter_out_npu_nocheck(self_ex, self_ex, dim, index, src);

    if (self_ex.scalar_type() != self_type) {
        self.copy_(acl_op::npu_dtype_cast(self_ex, self_type));
    }

    return self;
}
} // namespace

at::Tensor &scatter_out(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src,
                        at::Tensor &result)
{
    npu_preparation::CheckOut({self, src, index}, result, self);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        scatter_out_nocheck(contiguous_result, self, dim, index, src);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        scatter_out_nocheck(result, self, dim, index, src);
    }
    return result;
}

at::Tensor &scatter_out(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar &value,
                        at::Tensor &result)
{
    npu_preparation::CheckOut({self, index}, result, self);
    at::Tensor src_tensor = scalar_to_tensor(value).to(at::ScalarType::Float);
    src_tensor = npu_preparation::copy_tensor_host_to_device(src_tensor);
    at::Tensor src_tensor_broadcast = acl_op::npu_broadcast(src_tensor, op_infer::array_to_small_vector(index.sizes()));

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        scatter_out_nocheck(contiguous_result, self, dim, index, src_tensor_broadcast);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        scatter_out_nocheck(result, self, dim, index, src_tensor_broadcast);
    }
    return result;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index_ex, const at::Tensor &src)
{
    npu_preparation::check_memory({self, index_ex, src}, {self});

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        scatter_inplace_nocheck(contiguous_self, dim, index_ex, src);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        scatter_inplace_nocheck(self, dim, index_ex, src);
    }
    return self;
}

at::Tensor &scatter_(at::Tensor &self, int64_t dim, const at::Tensor &index_ex, const at::Scalar &src)
{
    npu_preparation::check_memory({self, index_ex}, {self});

    at::Tensor src_tensor = npu_preparation::copy_scalar_to_device(src, self.scalar_type());
    at::Tensor src_tensor_broadcast =
        acl_op::npu_broadcast(src_tensor, op_infer::array_to_small_vector(index_ex.sizes()));

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        scatter_inplace_nocheck(contiguous_self, dim, index_ex, src_tensor_broadcast);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        scatter_inplace_nocheck(self, dim, index_ex, src_tensor_broadcast);
    }
    return self;
}

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor scatter(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Tensor &src)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    acl_op::scatter_out(self, dim, index, src, result);
    return result;
}

at::Tensor scatter(const at::Tensor &self, int64_t dim, const at::Tensor &index, const at::Scalar &value)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    acl_op::scatter_out(self, dim, index, value, result);
    return result;
}
#endif

} // namespace acl_op
