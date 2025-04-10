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

at::Tensor& logsumexp_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnLogSumExp, acl_op::logsumexp_out(self, dims, keepdim, result));
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);
    EXEC_NPU_CMD(aclnnLogSumExp, self, dims, keepdim, result);
    return result;
}

at::Tensor& logsumexp_out(const at::Tensor& self, at::DimnameList dims, bool keepdim, at::Tensor& result)
{
    return op_api::logsumexp_out(self, dimnames_to_positions(self, dims), keepdim, result);
}

at::Tensor logsumexp(const at::Tensor& self, at::IntArrayRef dims, bool keepdim)
{
    DO_COMPATIBILITY(aclnnLogSumExp, acl_op::logsumexp(self, dims, keepdim));
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    at::ScalarType dst_type = self.scalar_type();
    if (isIntegralType(self.scalar_type(), true)) {
        dst_type = at::kFloat;
    }
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(outputSize, self.options().dtype(dst_type));
    EXEC_NPU_CMD(aclnnLogSumExp, self, dims, keepdim, result);

    return result;
}

at::Tensor logsumexp(const at::Tensor& self, at::DimnameList dims, bool keepdim)
{
    return op_api::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

}
