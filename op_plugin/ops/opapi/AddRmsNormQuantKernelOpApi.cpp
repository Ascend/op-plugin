// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
    using namespace op_plugin::utils;
    using namespace op_infer;

    tensor_list npu_add_rms_norm_quant(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma,
                                       const at::Tensor &scales1, const c10::optional<at::Tensor> &zero_points1,
                                       const c10::optional<at::Tensor> &beta, const c10::optional<at::Tensor> &scales2,
                                       const c10::optional<at::Tensor> &zero_points2, int64_t axis, double epsilon,
                                       bool div_mode, c10::optional<int64_t> dst_type)
    {
        TORCH_CHECK(!scales2.has_value(), "scales2 only support None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(!zero_points2.has_value(), "zero_points2 only support None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(axis == -1, "axis only support -1.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(div_mode == true, "div_mode only support True.", OPS_ERROR(ErrCode::PARAM));

        int64_t dst_type_value = dst_type.has_value() ? dst_type.value() : static_cast<int>(at::ScalarType::Char);
        at::Tensor y;
        aclDataType y_acltype = c10_npu::GetAclDataType(dst_type_value);
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);

        auto output_size_0 = x1.sizes();
        auto output_dtype_1 = x1.scalar_type();
        at::Tensor y1 = npu_preparation::apply_tensor_without_format(output_size_0, c10::dtype(scalar_dtype));
        at::Tensor y2 = npu_preparation::apply_tensor_without_format(output_size_0, c10::dtype(scalar_dtype));
        at::Tensor x_out = npu_preparation::apply_tensor_without_format(output_size_0, x1.options().dtype(output_dtype_1));
        at::Tensor rmsnorm_out{nullptr};

        TensorWrapper y1_wrapper = {y1, y_acltype};
        TensorWrapper y2_wrapper = {y2, y_acltype};
        if ((c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend910_95) && check_aclnn_kernel_available("aclnnAddRmsNormQuantV2")) {
            EXEC_NPU_CMD(aclnnAddRmsNormQuantV2, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, axis, epsilon, div_mode, y1, y2, x_out, rmsnorm_out);
        } else {
            TORCH_CHECK(!beta.has_value(), "In the current CANN version, aclnnAddRmsNormQuant does not support the parameter beta input. It is recommended to upgrade the CANN package. Or please remove the beta input parameter.", OPS_ERROR(ErrCode::PARAM));
            EXEC_NPU_CMD(aclnnAddRmsNormQuant, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode, y1_wrapper, y2_wrapper, x_out);
        }
        return std::make_tuple(std::move(y1), std::move(y2), std::move(x_out));
    }
}