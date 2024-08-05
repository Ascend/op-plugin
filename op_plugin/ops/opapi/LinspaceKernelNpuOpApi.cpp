// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
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

at::Tensor linspace(const at::Scalar& start, const at::Scalar& end, int64_t steps,
                    c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                    c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt)
{
    DO_COMPATIBILITY(aclnnLinspace, acl_op::linspace(start, end, steps, dtype_opt, layout_opt, device_opt, pin_memory_opt));
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative", OPS_ERROR(ErrCode::VALUE));
    c10::TensorOptions option =
      c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);
    at::SmallVector<int64_t, op_infer::SIZE> output_size = {steps};
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, option);
    EXEC_NPU_CMD(aclnnLinspace, start, end, steps, result);
    return result;
}

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnLinspace, acl_op::linspace_out(start, end, steps, result));
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative", OPS_ERROR(ErrCode::VALUE));

    if (result.numel() != steps) {
        result.resize_({steps});
    }

    EXEC_NPU_CMD(aclnnLinspace, start, end, steps, result);
    return result;
}

}
