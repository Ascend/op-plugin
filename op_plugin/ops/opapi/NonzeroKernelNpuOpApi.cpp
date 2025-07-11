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

at::Tensor exec_aclnn_non_zero(const at::Tensor& self, at::Tensor& out)
{
    static auto opApiFuncAddr = []() {
        auto ret = GetOpApiFuncAddr("aclGetViewShape");
        TORCH_CHECK(ret != nullptr, OPS_ERROR(ErrCode::PTR));
        return ret;
    }();
    using aclGetViewShapeFunc = int (*)(const aclTensor *tensor, int64_t **view_dims, uint64_t *view_dims_num);
    auto aclGetViewShape = reinterpret_cast<aclGetViewShapeFunc>(opApiFuncAddr);
    OP_EXEC_LOG(aclnnNonzero, "EXEC_NPU_CMD_SYNC", self, out);
    auto npuAclParams = EXEC_NPU_CMD_SYNC(aclnnNonzero, self, out);
    int64_t *view_dims = nullptr;
    uint64_t view_dim_num = 0;
    auto ret = aclGetViewShape(npuAclParams.Get<1>(), &view_dims, &view_dim_num);
    TORCH_CHECK(ret == 0, "aclGetViewShape failed.", OPS_ERROR(ErrCode::ACL));
    c10::SmallVector<int64_t, op_infer::SIZE> output_size(view_dims, view_dims + view_dim_num);
    out = out.resize_(output_size);
    // Need to use delete[] to release memory to avoid memory leakage!
    delete[] view_dims;
    view_dims = nullptr;
    return out;
}

at::Tensor& nonzero_out(const at::Tensor& self, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnNonzero, acl_op::nonzero_out(self, result));
    auto out_size = op_infer::nonzero_npu_max_output_size(self);
    at_npu::native::OpPreparation::check_tensor({self}, result, at::ScalarType::Long, out_size);
    auto out = result.is_contiguous() ? result : result.contiguous();
    auto output = exec_aclnn_non_zero(self, out);
    if (!result.is_contiguous()) {
        result.copy_(output);
    }
    return result;
}

at::Tensor nonzero(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnNonzero, acl_op::nonzero(self));
    auto out_size = op_infer::nonzero_npu_max_output_size(self);
    at::Tensor out =
        at_npu::native::OpPreparation::apply_tensor_without_format(out_size, self.options().dtype(at::kLong));
    return exec_aclnn_non_zero(self, out);
}

}
