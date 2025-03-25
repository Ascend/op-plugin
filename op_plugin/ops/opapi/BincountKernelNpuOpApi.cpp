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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor bincount(const at::Tensor& self, const c10::optional<at::Tensor>& weight, int64_t minlength)
{
    DO_COMPATIBILITY(aclnnBincount, acl_op::bincount(self, weight, minlength));
    // null tensor
    if (self.dim() == 1 && self.numel() == 0) {
        at::Tensor result;
        if (minlength <= 0) {
            result = npu_preparation::apply_tensor_without_format({0}, self.options().dtype(at::ScalarType::Long));
        } else {
            result = npu_preparation::apply_tensor_without_format({minlength}, self.options().dtype(at::ScalarType::Long));
            EXEC_NPU_CMD(aclnnBincount, self, weight, minlength, result);
        }
        return result;
    }

    // cheack non-negative
    auto min_value = op_api::min(self).item().toLong();
    TORCH_CHECK(min_value >= 0, "bincount only support 1-d non-negative integral inputs.", OPS_ERROR(ErrCode::VALUE));

    // calculate output size
    auto sizes = op_api::max(self).item().toLong();
    sizes = (sizes < minlength) ? minlength : (sizes + 1);

    // weight convert dtype as same as output defined by torch
    at::Tensor result;
    if (!weight.has_value()) {
        result = npu_preparation::apply_tensor_without_format({sizes}, self.options().dtype(at::ScalarType::Long));
    } else if (weight->dtype() == at::ScalarType::Float) {
        result = npu_preparation::apply_tensor_without_format({sizes}, weight->options().dtype(at::ScalarType::Float));
    } else {
        result = npu_preparation::apply_tensor_without_format({sizes}, weight->options().dtype(at::ScalarType::Double));
    }

    EXEC_NPU_CMD(aclnnBincount, self, weight, minlength, result);

    return result;
}
}
