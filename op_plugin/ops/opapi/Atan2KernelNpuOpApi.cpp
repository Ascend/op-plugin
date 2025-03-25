// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

at::Tensor& atan2_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &out)
{
    DO_COMPATIBILITY(aclnnAtan2, acl_op::atan2_out(self, other, out));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    npu_preparation::check_tensor({self, other}, out, out.scalar_type(), output_size);
    EXEC_NPU_CMD(aclnnAtan2, self, other, out);
    return out;
}

at::Tensor atan2(const at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnAtan2, acl_op::atan2(self, other));
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    c10::ScalarType infer_dtype = at::native::result_type(self, other);
    auto out_dtype = infer_dtype;
    if (isIntegralType(infer_dtype, true)) {
        out_dtype = at::kFloat;
    }
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_dtype));
    EXEC_NPU_CMD(aclnnAtan2, self, other, out);
    return out;
}

at::Tensor& atan2_(at::Tensor &self, const at::Tensor &other)
{
    DO_COMPATIBILITY(aclnnInplaceAtan2, acl_op::atan2_(self, other));
    EXEC_NPU_CMD(aclnnInplaceAtan2, self, other);
    return self;
}

}
