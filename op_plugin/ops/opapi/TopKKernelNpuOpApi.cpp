// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor&, at::Tensor&> topk_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    at::Tensor& values,
    at::Tensor& indices)
{
    if (self.dim() == 0 && k == 0) {
        TORCH_CHECK(values.dtype() == at::ScalarType::Long, "Expected out tensor to have dtype long int, bug got ",
                    values.dtype(), "instead.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(indices.dtype() == at::ScalarType::Long, "Expected out tensor to have dtype long int, bug got ",
                    indices.dtype(), "instead.", OPS_ERROR(ErrCode::PARAM));
        values.resize_as_(self).fill_(self);
        indices.resize_as_(self).fill_(0);
        return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
    }
    DO_COMPATIBILITY(aclnnTopk, acl_op::topk_out(self, k, dim, largest, sorted, values, indices));
    auto output_size = op_infer::topk_npu_output_size(self, k, dim);
    npu_preparation::check_tensor({self}, values, self.scalar_type(), output_size);
    npu_preparation::check_tensor({self}, indices, at::ScalarType::Long, output_size);

    EXEC_NPU_CMD(aclnnTopk, self, k, dim, largest, sorted, values, indices);
    return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted)
{
    DO_COMPATIBILITY(aclnnTopk, acl_op::topk(self, k, dim, largest, sorted));
    auto output_size = op_infer::topk_npu_output_size(self, k, dim);
    at::Tensor values;
    at::Tensor indices;
    if (self.dim() == 0 && k == 0) {
        values = at::zeros_like(self).fill_(self);
        indices = at::zeros_like(self);
        return std::tuple<at::Tensor, at::Tensor>(values, indices);
    }
    values = npu_preparation::apply_tensor_without_format(output_size, self.options());
    indices = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(at::kLong));

    EXEC_NPU_CMD(aclnnTopk, self, k, dim, largest, sorted, values, indices);
    return std::tuple<at::Tensor, at::Tensor>(values, indices);
}

} // namespace op_api
