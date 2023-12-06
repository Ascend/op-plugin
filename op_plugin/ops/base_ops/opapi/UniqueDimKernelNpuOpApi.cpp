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
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim(const at::Tensor &self, const int64_t dim, const bool sorted,
                                                          const bool return_inverse, const bool return_counts)
{
    DO_COMPATIBILITY(aclnnUniqueDim, acl_op::unique_dim(self, dim, sorted, return_inverse, return_counts));
    TORCH_CHECK(dim < self.dim(), "Dim's value must be smaller than self's dim.");
    at::Tensor y = npu_preparation::apply_tensor_without_format(self);
    at::Tensor y_inverse = npu_preparation::apply_tensor_without_format(self.size(dim), self.options().dtype(at::kLong));
    at::Tensor y_counts = npu_preparation::apply_tensor_without_format(self.size(dim), self.options().dtype(at::kLong));

    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, "GetOpApiFuncAddr failed.");
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnUniqueDim, self, sorted, return_inverse, dim, y,
                                          y_inverse, y_counts);
                                          
    int64_t *view_dims = nullptr;
    uint64_t view_dim_num = 0;
    constexpr int64_t Y_IDX = 4;
    auto ret1 = aclGetViewShape(npuAclParams.Get<Y_IDX>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret1 == 0, "aclGetViewShape for y failed.");
    c10::SmallVector<int64_t, SIZE> output_size_y(view_dims, view_dims + view_dim_num);
    y.resize_(output_size_y);

    constexpr int64_t Y_COUNTS_IDX = 6;
    auto ret2 = aclGetViewShape(npuAclParams.Get<Y_COUNTS_IDX>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret2 == 0, "aclGetViewShape for y_counts failed.");
    c10::SmallVector<int64_t, SIZE> output_size_y_counts(view_dims, view_dims + view_dim_num);
    y_counts.resize_(output_size_y_counts);

    delete view_dims;
    view_dims = nullptr;
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, y_inverse, y_counts);
}

} // namespace op_api
