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

at::Tensor& exp2_out(const at::Tensor& self, at::Tensor& out) {

    DO_COMPATIBILITY(aclnnExp2, acl_op::exp2_out(self, out));
    at_npu::native::OpPreparation::check_tensor({self}, out, out.scalar_type(), self.sizes());

    EXEC_NPU_CMD(aclnnExp2, self, out);

    return out;
}

at::Tensor exp2(const at::Tensor& self) {

    DO_COMPATIBILITY(aclnnExp2, acl_op::exp2(self));

    auto out_Dtype = self.dtype();
    if (isIntegralType(self.scalar_type(), true)) {
        out_Dtype = at::ScalarType::Float;
    }

    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(self.sizes(),
                     self.options().dtype(out_Dtype));

    EXEC_NPU_CMD(aclnnExp2, self, out);
    return out;
}

at::Tensor& exp2_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceExp2, acl_op::exp2_(self));
  EXEC_NPU_CMD(aclnnInplaceExp2, self);
  return self;
}

}  // namespace op_api

