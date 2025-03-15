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

at::Tensor &sum_out_common_nocheck(const at::Tensor &self,
                                   at::IntArrayRef dim,
                                   bool keepdim,
                                   c10::optional<c10::ScalarType> dtype,
                                   at::Tensor &result)
{
    DO_COMPATIBILITY(aclnnReduceSum, acl_op::sum_out(self, dim, keepdim, dtype, result));
    auto output_size = op_infer::sum_npu_output_size(self, dim, keepdim);
    auto res_type = dtype.has_value() ? dtype.value() : result.scalar_type();
    npu_preparation::check_tensor({self}, result, res_type, output_size);

    EXEC_NPU_CMD(aclnnReduceSum, self, dim, keepdim, res_type, result);
    return result;
}

at::Tensor sum_common_nocheck(const at::Tensor &self,
                              at::IntArrayRef dim,
                              bool keepdim,
                              c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnReduceSum, acl_op::sum(self, dim, keepdim, dtype));
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    auto self_size = self.sizes();
    auto out_type = self.scalar_type();

    if (dtype.has_value()) {
        out_type = dtype.value();
    } else if (isIntegralType(out_type, true)) {
        out_type = at::kLong;
    }

    for (uint64_t i = 0; i < self_size.size(); i++) {
        if (self_size[i] == 0) {
            return at::zeros(output_size, self.options().dtype(out_type));
        }
    }

    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_type));
    EXEC_NPU_CMD(aclnnReduceSum, self, dim, keepdim, out_type, result);
    return result;
}
} // namespace op_api
