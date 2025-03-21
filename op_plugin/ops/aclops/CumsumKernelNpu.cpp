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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &cumsum_out_nocheck(at::Tensor &result, const at::Tensor &self, int64_t dim)
{
    at::NoNamesGuard guard;
    at_npu::native::OpCommand cmd;
    // if dim = 0, performance in Aicpu is better than Aicore
    // if dim > INT32_MAX, we should use long to store dim for ensuring function correctness.
    // use host memory instead of scalar to improve delivery performance
    at::Scalar dimScalar(dim);
    cmd.Name("Cumsum").Input(self);
    if (dim == 0 || dim > INT32_MAX) {
        cmd.Input(dimScalar, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT);
    } else {
        cmd.Input(dimScalar, at::kInt, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT);
    }
    cmd.Output(result).Run();
    at::namedinference::propagate_names(result, self);
    return result;
}
} // namespace

at::Tensor &cumsum_out(const at::Tensor &self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor &result)
{
    at::ScalarType dst_type = self.scalar_type();
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (result.defined()) {
        dst_type = result.scalar_type();
    }
    at::Tensor self_cp =
        self.scalar_type() == dst_type ? self : at_npu::native::custom_ops::npu_dtype_cast(self, dst_type);
    npu_preparation::CheckOut({self_cp}, result, npu_preparation::get_tensor_npu_format(result), dst_type,
                              self_cp.sizes());
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        cumsum_out_nocheck(contiguous_result, self_cp, dim);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        cumsum_out_nocheck(result, self_cp, dim);
    }
    return result;
}

at::Tensor &cumsum_out(const at::Tensor &self, at::Dimname dim, c10::optional<at::ScalarType> dtype, at::Tensor &result)
{
    return acl_op::cumsum_out(self, dimname_to_position(self, dim), dtype, result);
}

at::Tensor cumsum(const at::Tensor &self, int64_t dim, const c10::optional<at::ScalarType> dtype)
{
    at::Tensor result;
    if (dtype.has_value()) {
        if (dtype.value() == at::kDouble) {
            TORCH_NPU_WARN_ONCE("[Cumsum] Dtype Double will be replaced with Float!");
            result = npu_preparation::apply_tensor(self, self.options().dtype(at::kFloat));
            return acl_op::cumsum_out(self, dim, at::kFloat, result);
        }
        result = npu_preparation::apply_tensor(self, self.options().dtype(dtype.value()));
    } else if (self.scalar_type() == at::ScalarType::Bool) {
        result = npu_preparation::apply_tensor(self, self.options().dtype(at::kLong));
    } else {
        result = npu_preparation::apply_tensor(self);
    }
    return acl_op::cumsum_out(self, dim, dtype, result);
}

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& cumsum_(
    at::Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype)
{
    return acl_op::cumsum_out(self, dim, dtype, self);
}
#endif
} // namespace acl_op
