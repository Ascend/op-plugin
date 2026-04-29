// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
at::Tensor bucketize(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right)
{
    TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(",
        boundaries.dim(), ")" + OPS_ERROR(ErrCode::PARAM));

    static bool isRegBaseSoc = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
    bool typeSupport = self.dtype() != at::kDouble && boundaries.dtype() != at::kDouble;
    static const bool is_bucketize_available = check_aclnn_kernel_available("aclnnBucketize");
    if (isRegBaseSoc && typeSupport && is_bucketize_available) {
        at::ScalarType expectedType = out_int32 ? at::kInt : at::kLong;
        at::TensorOptions opts = self.options().dtype(expectedType);
        at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(), opts);
        EXEC_NPU_CMD(aclnnBucketize, self, boundaries, out_int32, right, result);
        return result;
    } else {
        return op_api::searchsorted(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt);
    }
}

at::Tensor bucketize(const at::Scalar& self, const at::Tensor& boundaries, bool out_int32, bool right)
{
    TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(),
        ")" + OPS_ERROR(ErrCode::PARAM));
    return op_api::searchsorted(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt);
}

at::Tensor &bucketize_out(
    const at::Tensor& self,
    const at::Tensor& boundaries,
    bool out_int32,
    bool right,
    at::Tensor& out)
{
    TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(),
        ")" + OPS_ERROR(ErrCode::PARAM));

    static bool isRegBaseSoc = c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950;
    bool typeSupport = self.dtype() != at::kDouble && boundaries.dtype() != at::kDouble;
    static const bool is_bucketize_available = check_aclnn_kernel_available("aclnnBucketize");
    if (isRegBaseSoc && typeSupport && is_bucketize_available) {
        at_npu::native::OpPreparation::check_tensor({self, boundaries},
                                                    out,
                                                    out.scalar_type(),
                                                    self.sizes());
        EXEC_NPU_CMD(aclnnBucketize, self, boundaries, out_int32, right, out);
        return out;
    } else {
        return op_api::searchsorted_out(boundaries, self, out_int32, right, c10::nullopt, c10::nullopt, out);
    }
}
}
