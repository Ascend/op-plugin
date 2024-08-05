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
#include "torch_npu/csrc/framework/utils/UtilForOpAdapter.h"

namespace acl_op {
using npu_op_command = at_npu::native::OpCommand;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
static const int64_t FLOAT_STATUS_OP_DIMS_SIZE = 8;
const c10::SmallVector<int64_t, SIZE> output_size = {FLOAT_STATUS_OP_DIMS_SIZE};
enum CheckOverflowMode {
    OVERFLOW_ORIGINAL_MODE = 0,
    OVERFLOW_DEBUG_MODE = 1,
    OVERFLOW_UNDEFINED_MODE
};
} // namespace

at::Tensor npu_alloc_float_status(const at::Tensor& self) {
  auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(at::kFloat);
  at::Tensor result = npu_preparation::apply_tensor_with_format(
      output_size, options, npu_preparation::get_tensor_npu_format(self));
  npu_op_command cmd;
  cmd.Name("NPUAllocFloatStatus")
      .Output(result)
      .Run();
  return result;
}

at::Tensor npu_get_float_status(const at::Tensor& self, int64_t mode) {
    TORCH_CHECK((mode >= OVERFLOW_ORIGINAL_MODE && mode < OVERFLOW_UNDEFINED_MODE),
        "mode only supported for ORIGINAL_MODE(0) DEBUG_MODE(1), but got mode:", mode,
        OPS_ERROR(ErrCode::PARAM));

    npu_op_command cmd;
    if (mode == OVERFLOW_ORIGINAL_MODE) {
        if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
            at::Tensor out_tensor = npu_preparation::apply_tensor_with_format(
                output_size, self.options().dtype(at::kInt), npu_preparation::get_tensor_npu_format(self));
            cmd.Name("NPUGetFloatStatusV2")
                .Output(out_tensor)
                .Run();
            return out_tensor;
        } else {
            at::Tensor out_tensor = npu_preparation::apply_tensor(self, output_size);
            cmd.Name("NPUGetFloatStatus")
                .Input(self)
                .Output(out_tensor)
                .Run();
            return self;
        }
    } else {
        at::Tensor out_tensor = npu_preparation::apply_tensor_with_format(
            output_size, self.options().dtype(at::kInt), npu_preparation::get_tensor_npu_format(self));
        cmd.Name("NPUGetFloatDebugStatus")
            .Output(out_tensor)
            .Run();
       return out_tensor;
    }
}

at::Tensor npu_clear_float_status(const at::Tensor& self, int64_t mode) {
    TORCH_CHECK((mode >= OVERFLOW_ORIGINAL_MODE && mode < OVERFLOW_UNDEFINED_MODE),
        "mode only supported for ORIGINAL_MODE(0) DEBUG_MODE(1), but got mode:", mode,
        OPS_ERROR(ErrCode::PARAM));

    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    npu_op_command cmd;
    if (mode == OVERFLOW_ORIGINAL_MODE) {
        if (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1) {
            cmd.Name("NPUClearFloatStatusV2")
                .Run();
        } else {
            cmd.Name("NPUClearFloatStatus")
                .Input(self)
                .Output(result)
                .Run();
        }
    } else {
        cmd.Name("NPUClearFloatDebugStatus")
            .Run();
    }
    return result;
}
} // namespace acl_op
