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
#include "op_plugin/OpInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V2R1)
std::tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts)
{
    DO_COMPATIBILITY(aclnnUnique2, acl_op::_unique2(self, sorted, return_inverse, return_counts));
    at::Tensor y = npu_preparation::apply_tensor_without_format(self, self.numel());
    at::Tensor y_inverse = (return_inverse || return_counts)
                               ? npu_preparation::apply_tensor_without_format(self.sizes(),
                                                                              self.options().dtype(at::kLong))
                               : npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::kLong));
    at::Tensor y_counts = return_counts ? npu_preparation::apply_tensor_without_format(self.numel(),
                                                                                       self.options().dtype(at::kLong))
                                        : npu_preparation::apply_tensor_without_format({0},
                                                                                       self.options().dtype(at::kLong));
    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, OPS_ERROR(ErrCode::PTR));
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    OP_EXEC_LOG(aclnnUnique2, "EXEC_NPU_CMD_SYNC", self, sorted, return_inverse, return_counts, y, y_inverse, y_counts);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnUnique2, self, sorted, return_inverse, return_counts, y, y_inverse,
                                          y_counts);
    int64_t *view_dims = nullptr;
    uint64_t view_dim_num = 0;
    auto ret = aclGetViewShape(npuAclParams.Get<4>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret == 0, "aclGetViewShape failed.", OPS_ERROR(ErrCode::ACL));
    c10::SmallVector<int64_t, SIZE> output_size(view_dims, view_dims + view_dim_num);
    y.resize_(output_size);
    if (return_counts) {
        y_counts.resize_(output_size);
    }
    // Need to use delete[] to release memory to avoid memory leakage!
    delete[] view_dims;
    view_dims = nullptr;
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, y_inverse, y_counts);
}
#endif


#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
std::tuple<at::Tensor, at::Tensor, at::Tensor> _unique2(
    const at::Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts)
{
    DO_COMPATIBILITY(aclnnUnique2, acl_op::_unique2(self, sorted, return_inverse, return_counts));
    at::Tensor y = npu_preparation::apply_tensor_without_format(self, self.numel());
    at::Tensor y_inverse = (return_inverse || return_counts)
                              ? npu_preparation::apply_tensor_without_format(self.sizes(),
                                                                             self.options().dtype(at::kLong))
                              : npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::kLong));
    at::Tensor y_counts = return_counts ? npu_preparation::apply_tensor_without_format(self.numel(),
                                                                                       self.options().dtype(at::kLong))
                                        : npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::kLong));
    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, OPS_ERROR(ErrCode::VALUE));
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor* tensor, int64_t** view_dims, uint64_t* view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    OP_EXEC_LOG(aclnnUnique2, "EXEC_NPU_CMD_SYNC", self, sorted, return_inverse, return_counts, y, y_inverse, y_counts);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnUnique2, self, sorted, return_inverse, return_counts, y, y_inverse,
                                          y_counts);
    int64_t* view_dims = nullptr;
    uint64_t view_dim_num = 0;
    auto ret = aclGetViewShape(npuAclParams.Get<4>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret == 0, "aclGetViewShape failed.", OPS_ERROR(ErrCode::VALUE));
    c10::SmallVector<int64_t, SIZE> output_size(view_dims, view_dims + view_dim_num);
    y.resize_(output_size);
    if (return_counts) {
        y_counts.resize_(output_size);
    }
    // Need to use delete[] to release memory to avoid memory leakage!
    delete[] view_dims;
    view_dims = nullptr;
    if (!return_inverse) {
        y_inverse = npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::kLong));
    }
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, y_inverse, y_counts);
}
#endif
} // namespace op_api
