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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& ones_out(at::IntArrayRef size, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnInplaceOne, acl_op::ones_out(size, out));
    out.resize_(size);
    EXEC_NPU_CMD(aclnnInplaceOne, out);
    return out;
}

at::Tensor ones(at::IntArrayRef size,
                c10::optional<at::ScalarType> dtype,
                c10::optional<at::Layout> layout,
                c10::optional<at::Device> device,
                c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnInplaceOne, acl_op::ones(size, dtype, layout,
        device, pin_memory));
    auto real_device = device_or_default(device);
    at::TensorOptions option = c10::TensorOptions().dtype(dtype)
        .layout(layout)
        .device(real_device)
        .pinned_memory(pin_memory);
    at::Tensor result = npu_preparation::apply_tensor_without_format(size, option);
    EXEC_NPU_CMD(aclnnInplaceOne, result);
    return result;
}

at::Tensor ones(at::IntArrayRef size,
                c10::optional<at::DimnameList> names,
                c10::optional<at::ScalarType> dtype,
                c10::optional<at::Layout> layout,
                c10::optional<at::Device> device,
                c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnInplaceOne, acl_op::ones(size, names, dtype, layout,
                                                   device, pin_memory));
    auto real_device = device_or_default(device);
    at::TensorOptions option = c10::TensorOptions().dtype(dtype)
        .layout(layout)
        .device(real_device)
        .pinned_memory(pin_memory);
    at::Tensor result = npu_preparation::apply_tensor_without_format(size, option);
    EXEC_NPU_CMD(aclnnInplaceOne, result);
    auto maybe_name = names.value_or(at::ArrayRef<at::Dimname>{});
    at::namedinference::propagate_names_if_nonempty(result, maybe_name);
    return result;
}
} // op_api
