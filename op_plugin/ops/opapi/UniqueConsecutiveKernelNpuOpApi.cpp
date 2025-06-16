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

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive(const at::Tensor &self, bool return_inverse,
                                                                  bool return_counts, c10::optional<int64_t> dim)
{
    DO_COMPATIBILITY(aclnnUniqueConsecutive, acl_op::unique_consecutive(self, return_inverse, return_counts, dim));
    at::Tensor y = dim.has_value() ? npu_preparation::apply_tensor_without_format(self) :
                                     npu_preparation::apply_tensor_without_format(self, self.numel());
    if (dim.has_value()) {
        TORCH_CHECK(dim.value() < self.dim(), "Dim's value must be smaller than self's dim.", OPS_ERROR(ErrCode::VALUE));
    }
    at::Tensor y_inverse =
        dim.has_value() ?
            npu_preparation::apply_tensor_without_format(self.size(dim.value()), self.options().dtype(at::kLong)) :
            npu_preparation::apply_tensor_without_format(self.sizes(), self.options().dtype(at::kLong));
    at::Tensor y_counts =
        dim.has_value() ?
            npu_preparation::apply_tensor_without_format(self.size(dim.value()), self.options().dtype(at::kLong)) :
            npu_preparation::apply_tensor_without_format(self.numel(), self.options().dtype(at::kLong));
    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, "GetOpApiFuncAddr failed.", OPS_ERROR(ErrCode::INTERNAL));
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    constexpr int64_t NoneN = 1000;
    int64_t dim_value = dim.has_value() ? dim.value() : NoneN;
    OP_EXEC_LOG(aclnnUniqueConsecutive, "EXEC_NPU_CMD_SYNC",
                self, return_inverse, return_counts, dim_value, y, y_inverse, y_counts);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnUniqueConsecutive, self, return_inverse, return_counts, dim_value, y,
                                          y_inverse, y_counts);
    int64_t *view_dims = nullptr;
    uint64_t view_dim_num = 0;
    constexpr int64_t Y_IDX = 4;
    auto ret1 = aclGetViewShape(npuAclParams.Get<Y_IDX>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret1 == 0, "aclGetViewShape for y failed.", OPS_ERROR(ErrCode::ACL));
    c10::SmallVector<int64_t, SIZE> output_size_y(view_dims, view_dims + view_dim_num);
    y.resize_(output_size_y);

    constexpr int64_t Y_COUNTS_IDX = 6;
    auto ret2 = aclGetViewShape(npuAclParams.Get<Y_COUNTS_IDX>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret2 == 0, "aclGetViewShape for y_counts failed.", OPS_ERROR(ErrCode::ACL));
    c10::SmallVector<int64_t, SIZE> output_size_y_counts(view_dims, view_dims + view_dim_num);
    y_counts.resize_(output_size_y_counts);

    // Need to use delete[] to release memory to avoid memory leakage!
    delete[] view_dims;
    view_dims = nullptr;
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, y_inverse, y_counts);
}

}
