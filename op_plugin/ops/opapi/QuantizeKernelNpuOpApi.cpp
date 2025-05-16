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

const int64_t INT4_NUMS_IN_INT32_SPACE = 8;

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

    TORCH_CHECK(dtype == at::ScalarType::Char || dtype == at::ScalarType::QUInt4x2,
                "dtype must be Int8 or Int4" + OPS_ERROR(ErrCode::TYPE));
    at::Tensor result;
    if (dtype == at::ScalarType::QUInt4x2) {
        auto output_shape = op_infer::array_to_small_vector(self.sizes());
        auto x_dim_num = self.dim();
        TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
                    "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_shape[x_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
        dtype = at::ScalarType::Int;
        result = npu_preparation::apply_tensor_without_format(output_shape, self.options().dtype(dtype));
    } else {
        result = at_npu::native::OpPreparation::apply_tensor(self, self.options().dtype(dtype));
    }

    const bool sqrt_mode = false;

    static const bool is_ascend_quant_V3_available = check_aclnn_kernel_available("aclnnAscendQuantV3");
    if (!is_ascend_quant_V3_available) {
        EXEC_NPU_CMD(aclnnAscendQuant, self, scales, zero_points_opt, sqrt_mode, "round", dtype, result);
    } else {
        axis = axis < -1 ? axis : -1;
        EXEC_NPU_CMD(aclnnAscendQuantV3, self, scales, zero_points_opt, sqrt_mode, "round", dtype, axis, result);
    }

    return result;
}
}
