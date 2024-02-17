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
using npu_utils = at_npu::native::NpuUtils;

at::Tensor nanmedian(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnNanMedian, acl_op::nanmedian(self));
    at::SmallVector<int64_t, op_infer::SIZE> dims = op_plugin::utils::get_dimlist_for_tensor(self);
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, false);
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, output_size);
    EXEC_NPU_CMD(aclnnNanMedian, self, result);
    return result;
}

std::tuple<at::Tensor, at::Tensor> nanmedian(const at::Tensor &self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnNanMedianDim, acl_op::nanmedian(self, dim, keepdim));
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    at::Tensor output = npu_preparation::apply_tensor_without_format(self, output_size);
    at::Tensor indices = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));
    EXEC_NPU_CMD(aclnnNanMedianDim, self, dim, keepdim, output, indices);
    return std::tie(output, indices);
}

}  // namespace op_api
