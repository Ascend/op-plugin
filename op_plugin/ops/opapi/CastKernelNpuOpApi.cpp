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

#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor npu_dtype_cast_impl_op_api(const at::Tensor& self, at::ScalarType dtype)
{
    if (self.dtype() == dtype) {
        return self.clone();
    }
    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(self.sizes(),
                                                                     self.options().dtype(dtype));

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnCast, self, dtype, result);

    return result;
}
} // namespace

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype)
{
    DO_COMPATIBILITY(aclnnCast, acl_op::npu_dtype_cast(self, dtype));
    return npu_dtype_cast_impl_op_api(self, dtype);
}

at::Tensor npu_dtype_cast_backward(const at::Tensor& grad, at::ScalarType dtype)
{
    grad.requires_grad_();
    return at_npu::native::custom_ops::npu_dtype_cast(grad, dtype);
}

}
