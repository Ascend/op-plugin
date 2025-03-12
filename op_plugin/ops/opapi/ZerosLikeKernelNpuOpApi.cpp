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

at::Tensor& zero_(at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnInplaceZero, acl_op::zero_(self));
    EXEC_NPU_CMD(aclnnInplaceZero, self);
    return self;
}

at::Tensor zeros_like(
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> optional_memory_format)
{
    DO_COMPATIBILITY(aclnnInplaceZero, acl_op::zeros_like(self, dtype_opt, layout_opt, device_opt,
                                                          pin_memory_opt, optional_memory_format));
    auto device = device_opt.has_value() ? device_opt.value() : self.device();
    if (!torch_npu::utils::is_npu(device)) {
        auto result = at::empty_like(self, dtype_opt, layout_opt, device_opt, pin_memory_opt, optional_memory_format);
        return result.fill_(0);
    }

    auto other_options = c10::TensorOptions().dtype(dtype_opt)
        .device(device_opt)
        .layout(layout_opt)
        .pinned_memory(pin_memory_opt);
    auto options = self.options().merge_in(other_options);
    at::Tensor result = npu_preparation::apply_tensor(self, options);
    EXEC_NPU_CMD(aclnnInplaceZero, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

}  // namespace op_api
