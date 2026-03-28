// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;

static bool is_nz_format(const at::Tensor &tensor)
{
  const torch_npu::NPUStorageDesc &tensor_desc =
      torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
  return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
         tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4 ||
         tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_16;
}
at::Tensor npu_quant_grouped_matmul_dequant(const at::Tensor &x, const at::Tensor &quantized_weight,
                                            const at::Tensor &weight_scale, const at::Tensor &group_list,
                                            const c10::optional<at::Tensor> &bias,
                                            const c10::optional<at::Tensor> &x_scale,
                                            const c10::optional<at::Tensor> &x_offset,
                                            const c10::optional<at::Tensor> &smooth_scale,
                                            c10::optional<c10::string_view> quant_mode)
{
    if (is_nz_format(quantized_weight)) {
        static const bool is_weight_nz_available =
            check_aclnn_kernel_available("aclnnQuantGroupedMatmulDequantWeightNZ");
        TORCH_CHECK(is_weight_nz_available,
                    "Get aclnnQuantGroupedMatmulDequantWeightNZ failed, "
                    "please upgrade CANN.",
                    OPS_ERROR(ErrCode::PARAM));
    }
    auto quant_mode_attr = quant_mode.has_value() ? const_cast<char *>(quant_mode.value().data()) : nullptr;
    auto trans = true;
    auto output_size_0 = {x.size(0), weight_scale.size(1)};
    auto output_dtype_0 = x.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                  x.options().dtype(output_dtype_0));
    if (is_nz_format(quantized_weight)) {
      EXEC_NPU_CMD(aclnnQuantGroupedMatmulDequantWeightNZ, x, quantized_weight, weight_scale, group_list,
                   bias, x_scale, x_offset, smooth_scale, quant_mode_attr, trans, out);
    } else {
      EXEC_NPU_CMD(aclnnQuantGroupedMatmulDequant, x, quantized_weight, weight_scale, group_list,
                   bias, x_scale, x_offset, smooth_scale, quant_mode_attr, trans, out);
    }
  return out;
}

}  // namespace op_api