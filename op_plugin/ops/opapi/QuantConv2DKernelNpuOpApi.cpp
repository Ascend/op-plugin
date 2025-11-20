// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/custom_dtype/Init.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace c10_npu;

constexpr int ATTRS_DIM = 2;
constexpr int TENSORS_DIM = 4;
constexpr int INPUT_H_INDEX = 2;
constexpr int INPUT_W_INDEX = 3;
constexpr int WEIGHT_W_INDEX = 3;

void CheckParams(const at::Tensor& input, const at::Tensor& weight,
                 c10::IntArrayRef strides, c10::IntArrayRef pads, c10::IntArrayRef dilations,
                 c10::optional<int64_t> input_dtype, c10::optional<int64_t> weight_dtype,
                 const c10::optional<at::Tensor>& offset)
{
    TORCH_CHECK((input_dtype.has_value() && weight_dtype.has_value()) ||
                (!input_dtype.has_value() && !weight_dtype.has_value()),
                "input_dtype and weight_dtype are only support both None or not None, but now they are different.",
                OPS_ERROR(ErrCode::PARAM));
    if (input_dtype.has_value()) {
        TORCH_CHECK(input_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8) &&
                    weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                    "input_dtype and weight_dtype are only support torch_npu.hifloat8, but got input_dtype: ",
                    c10_npu::CustomDataTypeToString(input_dtype.value()), " and weight_dtype: ",
                    c10_npu::CustomDataTypeToString(weight_dtype.value()), OPS_ERROR(ErrCode::PARAM));
    }
    if (input_dtype.has_value()) {
        TORCH_CHECK((input.scalar_type() == at::ScalarType::Byte || input.scalar_type() == at::ScalarType::Char) &&
            (weight.scalar_type() == at::ScalarType::Byte || weight.scalar_type() == at::ScalarType::Char),
            "input and weight tensor dtype must be torch.uint8 or torch.int8, ",
            "when input_dtype and weight_dtype is hifloat8, but got input tensor dtype: ",
            input.scalar_type(), " and weight tensor dtype: ",
            weight.scalar_type(), OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(input.dim() >= TENSORS_DIM, "input has to be more than 4D, but got Tensor of dimension ",
                input.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= TENSORS_DIM, "weight has to be more than 4D, but got Tensor of dimension ",
                weight.dim(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(strides.size() >= ATTRS_DIM, "stride has to contain more than 2 elements, but got ",
                strides.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(pads.size() >= ATTRS_DIM, "padding has to contain more than 2 elements, but got ",
                pads.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(dilations.size() >= ATTRS_DIM, "dilation has to contain more than 2 elements, but got ",
                dilations.size(), OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(strides[0] * strides[1] != 0, "Stride cannot contain 0", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!offset.has_value(), "offset must be None, check the input offset", OPS_ERROR(ErrCode::PARAM));
}

at::Tensor npu_quant_conv2d_out(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& scale,
                                c10::IntArrayRef strides, c10::IntArrayRef pads, c10::IntArrayRef dilations,
                                int64_t groups, int32_t offset_x, c10::string_view round_mode,
                                c10::optional<int64_t> input_dtype, c10::optional<int64_t> weight_dtype,
                                c10::optional<int64_t> output_dtype, const c10::optional<at::Tensor>& bias,
                                const c10::optional<at::Tensor>& offset)
{
    CheckParams(input, weight, strides, pads, dilations, input_dtype, weight_dtype, offset);
    std::string round_mode_str = std::string(round_mode);
    char *round_mode_ptr = const_cast<char *>(round_mode_str.c_str());

    int64_t n = input.size(0);
    int64_t h = input.size(INPUT_H_INDEX);
    int64_t w = input.size(INPUT_W_INDEX);
    int64_t co = weight.size(0);
    auto kernel_size = weight.sizes().slice(2);

    // Conv output shape calc. PadTop == PadBottom.
    int64_t ho = (h + 2 * pads[0] - dilations[0] * (kernel_size[0] - 1) - 1) / strides[0] + 1;
    int64_t wo = (w + 2 * pads[1] - dilations[1] * (kernel_size[1] - 1) - 1) / strides[1] + 1;

    TORCH_CHECK(ho > 0, "Ho has to be positive, but got ", ho);
    TORCH_CHECK(wo > 0, "Wo has to be positive, but got ", wo);

    c10::SmallVector<int64_t, SIZE> out_size = {n, co, ho, wo};

    c10::TensorOptions options;
    TORCH_CHECK(output_dtype.has_value(), "output_dtype can not be None", OPS_ERROR(ErrCode::TYPE));
    if (output_dtype.value() == static_cast<int64_t>(at::ScalarType::Half)) {
        options = input.options().dtype(at::kHalf);
    } else if (output_dtype.value() == static_cast<int64_t>(at::ScalarType::Float)) {
        options = input.options().dtype(at::kFloat);
    } else if (output_dtype.value() == static_cast<int64_t>(at::ScalarType::BFloat16)) {
        options = input.options().dtype(at::kBFloat16);
    } else if (output_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8)) {
        options = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(output_dtype.value()));
    } else {
        TORCH_CHECK(false,
            "output_dtype must be one of "
            "[torch.float16, torch.bfloat16, torch.float32, torch_npu.hifloat8], ",
            "but got output_dtype: ", c10_npu::CustomDataTypeToString(output_dtype.value()),
            OPS_ERROR(ErrCode::TYPE));
    }

    auto output = npu_preparation::apply_tensor_without_format(out_size, options);

    bool transposed = false; // transposed is not used in npu
    c10::IntArrayRef outputPadding; // outputPadding is not used in npu
    if (input_dtype.has_value()) {
        TensorWrapper input_wrapper = {input, c10_npu::GetAclDataType(input_dtype.value())};
        TensorWrapper weight_wrapper = {weight, c10_npu::GetAclDataType(weight_dtype.value())};
        TensorWrapper output_wrapper = {output, (output_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8)) ?
            c10_npu::GetAclDataType(output_dtype.value()) :
            npu_preparation::convert_to_acl_data_type(output.scalar_type())};
        EXEC_NPU_CMD(aclnnQuantConvolution, input_wrapper, weight_wrapper, bias, scale, offset, strides, pads, dilations,
            transposed, outputPadding, groups, offset_x, round_mode_ptr, output_wrapper);
    } else {
        EXEC_NPU_CMD(aclnnQuantConvolution, input, weight, bias, scale, offset, strides, pads, dilations, transposed,
                     outputPadding, groups, offset_x, round_mode_ptr, output);
    }
    return output;
}

at::Tensor npu_quant_conv2d(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& scale,
                            c10::IntArrayRef strides, c10::IntArrayRef pads, c10::IntArrayRef dilations,
                            int64_t groups, int64_t offset_x, c10::string_view round_mode,
                            c10::optional<int64_t> output_dtype, const c10::optional<at::Tensor>& bias,
                            const c10::optional<at::Tensor>& offset, c10::optional<int64_t> input_dtype,
                            c10::optional<int64_t> weight_dtype)
{
    // If aclnn interface is not implemented, call aclop
    DO_COMPATIBILITY(aclnnQuantConvolution,
                     acl_op::npu_quant_conv2d(input, weight, scale, strides, pads, dilations, groups, offset_x,
                                              round_mode, output_dtype, bias, offset, input_dtype, weight_dtype));

    if (c10_npu::IsAclnnOnly()) {
        return npu_quant_conv2d_out(input, weight, scale, strides, pads, dilations, groups, static_cast<int32_t>(offset_x),
                                    round_mode, input_dtype, weight_dtype, output_dtype, bias, offset);
    } else {
        // aclnn only support 910_95 currently
        TORCH_NPU_WARN("current soc not support aclnn");
        return acl_op::npu_quant_conv2d(input, weight, scale, strides, pads, dilations, groups, offset_x,
                                        round_mode, output_dtype, bias, offset, input_dtype, weight_dtype);
    }
}

} // namespace op_api