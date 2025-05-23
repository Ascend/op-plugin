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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& addcdiv_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                        const at::Scalar& value, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnAddcdiv, acl_op::addcdiv_out(self, tensor1, tensor2, value, result));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnAddcdiv, self, tensor1, tensor2, value, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_names);
    return result;
}

at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                   const at::Scalar& value)
{
    if (isIntegralType(tensor1.scalar_type(), true) && isIntegralType(tensor2.scalar_type(), true)) {
        TORCH_CHECK(
            false,
            "Integer division with addcdiv is no longer supported, and in a future  ",
            "release addcdiv will perform a true division of tensor1 and tensor2. ",
            "The historic addcdiv behavior can be implemented as ",
            "(input + value * torch.trunc(tensor1 / tensor2)).to(input.dtype) ",
            "for integer inputs and as ",
            "(input + value * tensor1 / tensor2) for float inputs. ",
            "The future addcdiv behavior is just the latter implementation: ",
            "(input + value * tensor1 / tensor2), for all dtypes.");
    }
    DO_COMPATIBILITY(aclnnAddcdiv, acl_op::addcdiv(self, tensor1, tensor2, value));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto input_size = op_infer::broadcast_ops_npu_output_size(self, tensor1);
    auto output_size = op_infer::broadcast_ops_npu_output_size(input_size, tensor2.sizes());
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);

    EXEC_NPU_CMD(aclnnAddcdiv, self, tensor1, tensor2, value, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_names);
    return result;
}

at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value)
{
    DO_COMPATIBILITY(aclnnInplaceAddcdiv, acl_op::addcdiv_(self, tensor1, tensor2, value));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnInplaceAddcdiv, self, tensor1, tensor2, value);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}
} // namespace op_api
