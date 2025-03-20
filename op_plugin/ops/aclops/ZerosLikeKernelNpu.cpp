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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& zeros_like_out_npu_nocheck(at::Tensor& result, const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ZerosLike")
        .Input(self)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor zeros_like(
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format)
{
    auto real_device = device.has_value() ? device.value() : self.device();
    if (!torch_npu::utils::is_npu(real_device)) {
        auto result = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
        return result.fill_(0);
    }

    auto other_options = c10::TensorOptions().dtype(dtype)
                                            .device(device)
                                            .layout(layout)
                                            .pinned_memory(pin_memory);
    auto options = self.options().merge_in(other_options);
    at::Tensor result = npu_preparation::apply_tensor(self, options);

    return acl_op::zero_(result);
}

at::Tensor& zero_(at::Tensor& self)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        zeros_like_out_npu_nocheck(contiguous_self, contiguous_self);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        zeros_like_out_npu_nocheck(self, self);
    }

    return self;
}
} // namespace acl_op
