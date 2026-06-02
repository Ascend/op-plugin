// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
const int64_t INT4_IN_INT32_NUM = 8;

c10::SmallVector<int64_t, SIZE> get_rstd_shape(const at::Tensor &self)
{
    c10::SmallVector<int64_t, SIZE> rstd_shape;
    for (int64_t i = 0; i < self.dim() - 1; i++) {
        rstd_shape.emplace_back(self.size(i));
    }
    rstd_shape.emplace_back(1);
    return rstd_shape;
}

std::tuple<at::Tensor, at::Tensor> npu_rms_norm_quant_v2(const at::Tensor &x, const at::Tensor &gamma,
    const at::Tensor &scale, const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &beta,
    double epsilon, bool div_mode, c10::optional<int64_t> dst_dtype) {
    // output
    at::Tensor y;
    at::Tensor rstd;

    bool output_rstd = x.requires_grad();

    if (!output_rstd) {
        rstd = npu_preparation::apply_tensor_without_format({0}, c10::dtype(at::ScalarType::Float));
    } else {
        auto rstd_shape = get_rstd_shape(x);
        rstd = npu_preparation::apply_tensor_with_format(rstd_shape, x.options().dtype(at::kFloat), ACL_FORMAT_ND);
    }

    at::ScalarType scalar_dtype = at::ScalarType::Undefined;
    aclDataType y_acltype = aclDataType::ACL_INT8;

    auto output_shape = op_infer::array_to_small_vector(x.sizes());
    auto x_dim_num = x.dim();
    int64_t dst_dtype_value = dst_dtype.has_value() ? dst_dtype.value() : static_cast<int>(at::ScalarType::Char);
    if (dst_dtype_value == static_cast<int64_t>(at::ScalarType::QUInt4x2)) {
        // int4 pack to int32
        ASCEND_LOGI("[npu_rms_norm_quant]: dst_dtype is torch.quint4x2, setting aclTensor out dtype to: %s",
            at_npu::native::AclDataTypeToString(aclDataType::ACL_INT32).c_str());
        y_acltype = aclDataType::ACL_INT32;
        scalar_dtype = at::ScalarType::Int;

        TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_IN_INT32_NUM == 0,
            "x shape last dim must be divded by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        output_shape[x_dim_num - 1] /= INT4_IN_INT32_NUM;
        int64_t npu_format = at_npu::native::custom_ops::get_npu_format(x);
        if (npu_format == ACL_FORMAT_FRACTAL_NZ) {
            y = npu_preparation::apply_tensor_with_format(
                output_shape, c10::dtype(scalar_dtype), ACL_FORMAT_FRACTAL_NZ, true);
        } else {
            y = npu_preparation::apply_tensor_without_format(output_shape, c10::dtype(scalar_dtype));
        }
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_dtype_value);
        scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        y = npu_preparation::apply_tensor_without_format(output_shape, c10::dtype(scalar_dtype));
    }

    TensorWrapper y_wrapper = {y, y_acltype};
    EXEC_NPU_CMD(aclnnRmsNormQuantV3, x, gamma, scale, offset, beta, epsilon, div_mode, output_rstd, y_wrapper, rstd);
    return std::tuple<at::Tensor, at::Tensor>(y, rstd);
}
}