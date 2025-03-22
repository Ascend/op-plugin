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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor &logical_or_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar &other)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("LogicalOr").Input(self).Input(other, self.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &logical_or_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    if (npu_preparation::IsCPUScalar(self)) {
        logical_or_out_npu_nocheck(result, other, self.item());
    } else if (npu_preparation::IsCPUScalar(other)) {
        logical_or_out_npu_nocheck(result, self, other.item());
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("LogicalOr").Input(self).Input(other).Output(result).Run();
    }
    return result;
}
} // namespace

at::Tensor &logical_or_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self, other}, out, npu_preparation::get_tensor_npu_format(self), out.scalar_type(),
                              output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        logical_or_out_npu_nocheck(contiguous_result, self, other);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        logical_or_out_npu_nocheck(out, self, other);
    }
    return out;
}

at::Tensor logical_or(const at::Tensor &self, const at::Tensor &other)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    logical_or_out_npu_nocheck(result, self, other);
    result = at_npu::native::custom_ops::npu_dtype_cast(result, at::kBool);
    return result;
}

at::Tensor &logical_or_(at::Tensor &self, const at::Tensor &other)
{
    return acl_op::logical_or_out(self, other, self);
}
} // namespace acl_op
