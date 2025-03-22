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
at::Tensor &logical_and_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Scalar other)
{
    auto self_copy = (self.dtype() == at::kBool) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool);
    at_npu::native::OpCommand cmd;
    cmd.Name("LogicalAnd").Input(self_copy).Input(other, self_copy.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &logical_and_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
    if (self.dim() == 0) {
        logical_and_out_npu_nocheck(result, other, self.item());
    } else if (other.dim() == 0) {
        logical_and_out_npu_nocheck(result, self, other.item());
    } else {
        auto self_copy =
            (self.dtype() == at::kBool) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool);
        auto other_copy =
            (other.dtype() == at::kBool) ? other : at_npu::native::custom_ops::npu_dtype_cast(other, at::kBool);

        at_npu::native::OpCommand cmd;
        cmd.Name("LogicalAnd").Input(self_copy).Input(other_copy).Output(result).Run();
    }
    return result;
}
} // namespace

at::Tensor &logical_and_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self}, out, npu_preparation::get_tensor_npu_format(self), out.scalar_type(),
                              output_size);

    if (npu_utils::check_match(&out) && (out.dtype() == at::kBool)) {
        logical_and_out_npu_nocheck(out, self, other);
    } else {
        auto result_copy = npu_preparation::ApplyTensorWithSizes(output_size, self.options().dtype(at::kBool));
        logical_and_out_npu_nocheck(result_copy, self, other);
        result_copy = at_npu::native::custom_ops::npu_dtype_cast(result_copy, self.scalar_type());
        npu_utils::format_fresh_view(out, result_copy);
    }
    return out;
}

at::Tensor logical_and(const at::Tensor &self, const at::Tensor &other)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kBool),
                                                                  npu_preparation::get_tensor_npu_format(self));
    logical_and_out_npu_nocheck(result, self, other);
    return result;
}

at::Tensor &logical_and_(at::Tensor &self, const at::Tensor &other)
{
    TORCH_CHECK(self.dtype() == other.dtype(), "Expected object of scalar type ", self.dtype(), " but got scalar type ",
        other.dtype(), " for argument 'other'" + OPS_ERROR(ErrCode::PARAM));
    return acl_op::logical_and_out(self, other, self);
}
} // namespace acl_op
