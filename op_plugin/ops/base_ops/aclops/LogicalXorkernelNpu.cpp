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
at::Tensor &logical_xor_out_npu_nocheck(const at::Tensor &self, const at::Scalar other, at::Tensor &result)
{
    auto self_copy = (self.dtype() == at::kBool) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool);
    at_npu::native::OpCommand cmd;
    cmd.Name("NotEqual").Input(self_copy).Input(other, self_copy.scalar_type()).Output(result).Run();
    return result;
}

at::Tensor &logical_xor_out_npu_nocheck(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    if (self.dim() == 0) {
        logical_xor_out_npu_nocheck(other, self.item(), result);
    } else if (other.dim() == 0) {
        logical_xor_out_npu_nocheck(self, other.item(), result);
    } else {
        auto selfCopy =
            (self.dtype() == at::kBool) ? self : at_npu::native::custom_ops::npu_dtype_cast(self, at::kBool);
        auto otherCopy =
            (other.dtype() == at::kBool) ? other : at_npu::native::custom_ops::npu_dtype_cast(other, at::kBool);

        at_npu::native::OpCommand cmd;
        cmd.Name("NotEqual").Input(selfCopy).Input(otherCopy).Output(result).Run();
    }
    return result;
}
} // namespace

at::Tensor &logical_xor_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
    auto outputSize = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::CheckOut({self}, result, npu_preparation::get_tensor_npu_format(self), result.scalar_type(),
                              outputSize);

    if (npu_utils::check_match(&result) && (result.dtype() == at::kBool)) {
        logical_xor_out_npu_nocheck(self, other, result);
    } else {
        auto result_copy = npu_preparation::apply_tensor_with_sizes(outputSize, self.options().dtype(at::kBool));
        logical_xor_out_npu_nocheck(self, other, result_copy);
        result_copy = at_npu::native::custom_ops::npu_dtype_cast(result_copy, result.scalar_type());
        npu_utils::format_fresh_view(result, result_copy);
    }
    return result;
}

at::Tensor logical_xor(const at::Tensor &self, const at::Tensor &other)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result = npu_preparation::apply_tensor_with_format(output_size, self.options().dtype(at::kBool),
                                                                  npu_preparation::get_tensor_npu_format(self));
    logical_xor_out_npu_nocheck(self, other, result);
    return result;
}

} // namespace acl_op
