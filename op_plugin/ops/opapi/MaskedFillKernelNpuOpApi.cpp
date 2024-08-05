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

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    auto maybe_outnames = at::namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
    DO_COMPATIBILITY(aclnnInplaceMaskedFillTensor, acl_op::masked_fill_(self, mask, value));
    if (at_npu::native::OpPreparation::IsCPUScalar(value)) {
        at::Scalar scalar = value.item();
        auto value_cp = at_npu::native::OpPreparation::copy_scalar_to_device(scalar, value.scalar_type(),
                                                                             self.device());
        EXEC_NPU_CMD(aclnnInplaceMaskedFillTensor, self, mask, value_cp);
    } else {
        EXEC_NPU_CMD(aclnnInplaceMaskedFillTensor, self, mask, value);
    }
    at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
    return self;
}

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value)
{
    auto maybe_outnames = at::namedinference::broadcast_to_outnames(self, mask, "masked_fill_");
    DO_COMPATIBILITY(aclnnInplaceMaskedFillScalar, acl_op::masked_fill_(self, mask, value));
    EXEC_NPU_CMD(aclnnInplaceMaskedFillScalar, self, mask, value);
    at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
    return self;
}

}
