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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
at::Tensor& prod_out(const at::Tensor& self, int64_t dim, bool keepdim,
                     c10::optional<at::ScalarType> dtype, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnProdDim, acl_op::prod_out(self, dim, keepdim, dtype, result));

    at::ScalarType dst_type = dtype.has_value() ? dtype.value() : result.scalar_type();
    // calculate the output size
    auto output_size = op_infer::prod_npu_output_size(self, dim, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, result, dst_type, output_size);
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnProdDim, self, dim, keepdim, dst_type, result);

    return result;
}

at::Tensor prod(const at::Tensor& self, int64_t dim,
                bool keepdim, c10::optional<at::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnProdDim, acl_op::prod(self, dim, keepdim, dtype));

    at::ScalarType dst_type = self.scalar_type();
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (isIntegralType(self.scalar_type(), true)) {
        dst_type = at::kLong;
    }
    // calculate the output size
    auto output_size = op_infer::prod_npu_output_size(self, dim, keepdim);
    // construct the output tensor of the NPU
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        self.options().dtype(dst_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnProdDim, self, dim, keepdim, dst_type, result);

    return result;
}

at::Tensor prod(const at::Tensor& self, c10::optional<at::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnProd, acl_op::prod(self, dtype));

    at::ScalarType dst_type = self.scalar_type();
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (isIntegralType(self.scalar_type(), true)) {
        dst_type = at::kLong;
    }
    // calculate the output size
    auto output_size = op_infer::prod_npu_output_size(self, false);
    // construct the output tensor of the NPU
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        self.options().dtype(dst_type));
    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnProd, self, dst_type, result);

    return result;
}
}
