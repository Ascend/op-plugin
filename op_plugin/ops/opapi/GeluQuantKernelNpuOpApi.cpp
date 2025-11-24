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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
namespace {
}; // namespace

std::tuple<at::Tensor, at::Tensor> npu_gelu_quant(
    const at::Tensor &self,
    const c10::optional<at::Tensor> &input_scale,
    const c10::optional<at::Tensor> &input_offset,
    c10::string_view approximate,
    c10::string_view quant_mode,
    c10::optional<int64_t> dst_type,
    c10::string_view round_mode)
{
    const at::Tensor &input_scale_value = c10::value_or_else(input_scale, [] { return at::Tensor(); });
    const at::Tensor &input_offset_value = c10::value_or_else(input_offset, [] { return at::Tensor(); });

    at::Tensor y;
    at::Tensor out_scale;
    aclDataType y_acltype;
    if (!dst_type.has_value()) {
        ASCEND_LOGI("[npu_gelu_quant]: Parameter(dst_type) is None, setting aclTensor y dtype to default: %s",
                    at_npu::native::AclDataTypeToString(aclDataType::ACL_INT8).c_str());
        y_acltype = aclDataType::ACL_INT8;
    } else {
        ASCEND_LOGI("[npu_gelu_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type.value());
        y_acltype = c10_npu::GetAclDataType(dst_type.value());
        ASCEND_LOGI("[npu_gelu_quant]: Setting aclTensor y dtype to: %s", at_npu::native::AclDataTypeToString(y_acltype).c_str());
    }
    at::ScalarType y_scalar_type = npu_preparation::convert_to_scalar_type(y_acltype);

    auto y_shape = op_infer::array_to_small_vector(self.sizes());
    y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(y_scalar_type));
    TensorWrapper y_wrapper = {y, y_acltype};

    std::string approximate_str = std::string(approximate.data());
    std::string quant_mode_str = std::string(quant_mode.data());
    std::string round_mode_str = std::string(round_mode.data());
    const char* approximate_char = approximate_str.c_str();
    const char* quant_mode_char = quant_mode_str.c_str();
    const char* round_mode_char = round_mode_str.c_str();
    bool quant_mode_valid = (quant_mode_str == "dynamic" || quant_mode_str == "static");
    TORCH_CHECK(quant_mode_valid,
                "Parameter quant_mode must be 'dynamic' or 'static', got: " + quant_mode_str + OPS_ERROR(ErrCode::PARAM));
    if (quant_mode_str == "dynamic") {
        at::SmallVector<int64_t, op_infer::SIZE> scale_shape = {};
        int scale_dim = self.dim() - 1;
        int index = 0;
        for (; index < scale_dim; ++index) {
            scale_shape.push_back(self.size(index));
        }
        out_scale = npu_preparation::apply_tensor_without_format(scale_shape, c10::dtype(c10::ScalarType::Float));
        TensorWrapper out_scale_wrapper = {out_scale, aclDataType::ACL_FLOAT};
        EXEC_NPU_CMD(aclnnGeluQuant, self, input_scale_value, input_offset_value,
                     approximate_char, quant_mode_char, round_mode_char, y_acltype, y_wrapper, out_scale_wrapper);
        return std::make_tuple(y, out_scale);
    } else {
        EXEC_NPU_CMD(aclnnGeluQuant, self, input_scale_value, input_offset_value,
                     approximate_char, quant_mode_char, round_mode_char, y_acltype, y_wrapper, out_scale);
        return std::make_tuple(y, out_scale);
    }
}

} // namespace op_api