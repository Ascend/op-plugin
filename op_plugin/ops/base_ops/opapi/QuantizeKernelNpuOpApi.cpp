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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_quantize(
    const at::Tensor& self,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& zero_points_opt,
    at::ScalarType dtype,
    int64_t axis,
    bool div_mode)
{
    if (div_mode) {
        return acl_op::npu_quantize(self, scales, zero_points_opt, dtype, axis);
    }
    if (dtype == at::kQInt8) {
        dtype = at::kChar;
    }
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor(self, self.options().dtype(dtype));
    const bool sqrt_mode = false;
    EXEC_NPU_CMD(aclnnAscendQuant, self, scales, zero_points_opt, sqrt_mode, "round", dtype, result);
    return result;
}
} // namespace op_api
