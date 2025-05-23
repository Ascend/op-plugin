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

#include <ATen/native/TypeProperties.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& addcmul_out(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                        const at::Scalar& value, at::Tensor& result)
{
    at::TensorList tensors = {self, tensor1, tensor2};
    auto high_type = at::native::result_type(tensors);
    at::ScalarType result_type = result.scalar_type();

    TORCH_CHECK(canCast(high_type, result_type), "result type ", high_type,
        " can't be cast to the desired output type ", result_type, OPS_ERROR(ErrCode::TYPE));

    DO_COMPATIBILITY(aclnnAddcmul, acl_op::addcmul_out(self, tensor1, tensor2, value, result));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto input_size = op_infer::broadcast_ops_npu_output_size(self, tensor1);
    auto output_size = op_infer::broadcast_ops_npu_output_size(input_size, tensor2.sizes());
    npu_preparation::check_tensor({self}, result, high_type, output_size);
    EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_names);
    return result;
}

at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                   const at::Scalar& value)
{
    at::TensorList tensors = {self, tensor1, tensor2};
    auto high_type = at::native::result_type(tensors);

    DO_COMPATIBILITY(aclnnAddcmul, acl_op::addcmul(self, tensor1, tensor2, value));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto input_size = op_infer::broadcast_ops_npu_output_size(self, tensor1);
    auto output_size = op_infer::broadcast_ops_npu_output_size(input_size, tensor2.sizes());
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(high_type));

    EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
    at::namedinference::propagate_names_if_nonempty(result, maybe_names);
    return result;
}


at::Tensor& addcmul_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2, const at::Scalar& value)
{
    at::TensorList tensors = {self, tensor1, tensor2};
    auto high_type = at::native::result_type(tensors);
    at::ScalarType self_type = self.scalar_type();

    TORCH_CHECK(canCast(high_type, self_type), "result type ", high_type,
        " can't be cast to the desired output type ", self_type, OPS_ERROR(ErrCode::TYPE));

    DO_COMPATIBILITY(aclnnInplaceAddcmul, acl_op::addcmul_(self, tensor1, tensor2, value));
    std::vector<at::Tensor> tensor_list = {self, tensor1, tensor2};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnInplaceAddcmul, self, tensor1, tensor2, value);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}

}  // namespace op_api
