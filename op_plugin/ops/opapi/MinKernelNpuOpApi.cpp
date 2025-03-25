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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor minimum(const at::Tensor& self, const at::Tensor& other)
{
    DO_COMPATIBILITY(aclnnMinimum, acl_op::minimum(self, other));
    auto result_type = at::result_type(self, other);
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(result_type));
    return op_api::minimum_out(self, other, result);
}

at::Tensor& minimum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnMinimum, acl_op::minimum_out(self, other, result));
    at::Tensor cp_other = other;
    at::Tensor cp_self = self;
    if (at_npu::native::OpPreparation::IsCPUScalar(other)) {
        at::Scalar scalar = other.item();
        cp_other = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, other.scalar_type(), self.device());
    } else if (at_npu::native::OpPreparation::IsCPUScalar(self)) {
        at::Scalar scalar = self.item();
        cp_self = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, self.scalar_type(), other.device());
    }
    auto output_size = op_infer::broadcast_ops_npu_output_size(cp_self, cp_other);
    at_npu::native::OpPreparation::check_tensor({cp_self, cp_other}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnMinimum, cp_self, cp_other, result);
    return result;
}


at::Tensor min(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnMin, acl_op::min(self));
    at::SmallVector<int64_t, op_infer::SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnMin, self, result);
    return result;
}

std::tuple<at::Tensor&, at::Tensor&> min_out(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices)
{
    DO_COMPATIBILITY(aclnnMinDim, acl_op::min_out(self, dim, keepdim, output, indices));
    at::SmallVector<int64_t, op_infer::SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, output, self.scalar_type(), output_size);
    at_npu::native::OpPreparation::check_tensor({self}, indices, at::ScalarType::Long, output_size);
    EXEC_NPU_CMD(aclnnMinDim, self, dim, keepdim, output, indices);
    return std::tie(output, indices);
}

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnMinDim, acl_op::min(self, dim, keepdim));
    at::SmallVector<int64_t, op_infer::SIZE> dims = {dim};
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    at::Tensor outputs = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
    at::Tensor indices = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size, self.options().dtype(at::ScalarType::Long));
    EXEC_NPU_CMD(aclnnMinDim, self, dim, keepdim, outputs, indices);
    return std::tie(outputs, indices);
}

at::Tensor& min_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnMinimum, acl_op::min_out(self, other, result));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at_npu::native::OpPreparation::check_tensor({self, other}, result, result.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnMinimum, self, other, result);
    return result;
}

std::tuple<at::Tensor&, at::Tensor&> min_out(
    const at::Tensor& self,
    at::Dimname dim,
    bool keepdim,
    at::Tensor& output,
    at::Tensor& indices)
{
    DO_COMPATIBILITY(aclnnMinDim, acl_op::min_out(self, dim, keepdim, output, indices));
    return op_api::min_out(self, dimname_to_position(self, dim), keepdim, output, indices);
}

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, at::Dimname dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnMinDim, acl_op::min(self, dim, keepdim));
    return op_api::min(self, dimname_to_position(self, dim), keepdim);
}

}
