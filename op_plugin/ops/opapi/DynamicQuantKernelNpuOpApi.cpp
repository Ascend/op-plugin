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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {

const int USE_ACLNN_DYNAMIC_QUANT_V1 = 1;
const int USE_ACLNN_DYNAMIC_QUANT_V2 = 2;
const int USE_ACLNN_DYNAMIC_QUANT_V3 = 3;
const int USE_ACLNN_DYNAMIC_QUANT_V4 = 4;

const int64_t INT4_IN_INT32_NUM = 8;
constexpr int64_t DTYPE_NUM_FOR_QUINT4X2 = static_cast<int64_t>(at::ScalarType::QUInt4x2);
constexpr int64_t INPUT_DIM_LOWER_BOUND = 1;

TensorWrapper get_output_tensor_wrapper(
    const at::Tensor &input, at::Tensor &output,
    aclDataType &y_acltype, c10::optional<int64_t> dst_type,
    at::SmallVector<int64_t, op_infer::SIZE> scale_size, int index)
{
    if (dst_type == DTYPE_NUM_FOR_QUINT4X2) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "Input shape last dim must be divded by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        at::SmallVector<int64_t, op_infer::SIZE> input_shape_copy(input.sizes());
        input_shape_copy[index] /= INT4_IN_INT32_NUM;
        output = npu_preparation::apply_tensor_without_format(input_shape_copy, c10::dtype(c10::ScalarType::Int));
        y_acltype = aclDataType::ACL_INT32;
    } else if (!dst_type.has_value()) { // 默认INT8
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
        y_acltype = aclDataType::ACL_INT8;
    } else {
        y_acltype = c10_npu::GetAclDataType(dst_type.value());
        at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(scalar_dtype));
    }
    TensorWrapper y_wrapper = {output, y_acltype};
    return y_wrapper;
}

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant_v0(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type)
{
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    at::Tensor output;
    aclDataType y_acltype;
    TensorWrapper y_wrapper = get_output_tensor_wrapper(input, output, y_acltype, dst_type, scale_size, index);
    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, y_wrapper, scale);
    return std::make_tuple(output, scale);
}

struct DynamicQuantParams {
    std::string quant_mode = "pertoken";
    bool is_symmetrical = true;
    aclDataType dst_type = aclDataType::ACL_INT8;
    float dst_type_max = 0.0;
};

int select_version(const DynamicQuantParams& attr)
{
    static bool npu_support_v3 = check_aclnn_kernel_available("aclnnDynamicQuantV3");
    static bool npu_support_v4 = check_aclnn_kernel_available("aclnnDynamicQuantV4");
    if (!npu_support_v3 && !npu_support_v4) {
        return USE_ACLNN_DYNAMIC_QUANT_V2;
    }

    if (attr.dst_type_max != 0.0) {
        TORCH_CHECK(npu_support_v4,
            "Can't support attr dst_type_max, please check CANN version." + OPS_ERROR(ErrCode::PARAM));
        return USE_ACLNN_DYNAMIC_QUANT_V4;
    }

    if (attr.quant_mode != "pertoken") {
        TORCH_CHECK(npu_support_v3,
            "Can't support attr quant_mode, please check CANN version." + OPS_ERROR(ErrCode::PARAM));
        return npu_support_v4 ? USE_ACLNN_DYNAMIC_QUANT_V4 : USE_ACLNN_DYNAMIC_QUANT_V3;
    }

    return USE_ACLNN_DYNAMIC_QUANT_V2;
}

template <typename T>
void dynamic_quant_run_aclnn(const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    TensorWrapper &y_wrapper,
    at::Tensor &scale,
    T &offset,
    const DynamicQuantParams& attr)
{
    int version = select_version(attr);
    const char* quant_mode = attr.quant_mode.c_str();
    switch (version) {
        case USE_ACLNN_DYNAMIC_QUANT_V4:
            EXEC_NPU_CMD(aclnnDynamicQuantV4, input, smooth_scales, group_index, attr.dst_type,
                attr.is_symmetrical, quant_mode, attr.dst_type_max, y_wrapper, scale, offset);
            break;
        case USE_ACLNN_DYNAMIC_QUANT_V3:
            EXEC_NPU_CMD(aclnnDynamicQuantV3, input, smooth_scales, group_index, attr.dst_type,
                attr.is_symmetrical, quant_mode, y_wrapper, scale, offset);
            break;
        case USE_ACLNN_DYNAMIC_QUANT_V2:
            EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index,
                attr.dst_type, y_wrapper, scale, offset);
            break;
        default:
            npu_dynamic_quant_v0(input, smooth_scales, group_index, attr.dst_type);
            break;
    }
}

} // namespace

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type,
    c10::string_view quant_mode,
    double dst_type_max)
{
    TORCH_CHECK(input.dim() > INPUT_DIM_LOWER_BOUND, "Input shape dim should be greater than 1" + OPS_ERROR(ErrCode::PARAM));
    // if aclnnDynamicQuantV2 is not implemented, use npu_dynamic_quant_v0(aclnnDynamicQuant)
    DO_COMPATIBILITY(aclnnDynamicQuantV2, npu_dynamic_quant_v0(input, smooth_scales, group_index, dst_type));

    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim - 1; ++index) {
        scale_size.push_back(input.size(index));
    }

    DynamicQuantParams attr;
    attr.quant_mode = std::string(quant_mode);
    attr.is_symmetrical = true;
    attr.dst_type_max = static_cast<float>(dst_type_max);

    if (attr.quant_mode == "perchannel") {
        scale_size.push_back(input.size(scale_dim));
    } else {
        scale_size.push_back(input.size(index));
    }
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    c10::optional<at::Tensor> offset;

    at::Tensor output;
    aclDataType y_acltype;
    TensorWrapper y_wrapper = get_output_tensor_wrapper(input, output, y_acltype, dst_type, scale_size, index + 1);
    attr.dst_type = y_acltype;

    if (attr.quant_mode == "pertensor") {
        at::SmallVector<int64_t, op_infer::SIZE> per_tensor_size = {1};
        scale = npu_preparation::apply_tensor_without_format(per_tensor_size, c10::dtype(c10::ScalarType::Float));
    }

    dynamic_quant_run_aclnn<c10::optional<at::Tensor>>(input, smooth_scales, group_index, y_wrapper, scale, offset, attr);

    return std::make_tuple(output, scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_quant_asymmetric(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<int64_t> dst_type,
    c10::string_view quant_mode,
    double dst_type_max)
{
    TORCH_CHECK(input.dim() > INPUT_DIM_LOWER_BOUND, "Input shape dim should be greater than 1" + OPS_ERROR(ErrCode::PARAM));
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim - 1; ++index) {
        scale_size.push_back(input.size(index));
    }

    DynamicQuantParams attr;
    attr.quant_mode = std::string(quant_mode);
    attr.is_symmetrical = false;
    attr.dst_type_max = static_cast<float>(dst_type_max);

    if (attr.quant_mode == "perchannel") {
        scale_size.push_back(input.size(scale_dim));
    } else {
        scale_size.push_back(input.size(index));
    }

    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    at::Tensor offset = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    at::Tensor output;
    aclDataType y_acltype;
    TensorWrapper y_wrapper = get_output_tensor_wrapper(input, output, y_acltype, dst_type, scale_size, index + 1);
    attr.dst_type = y_acltype;

    if (attr.quant_mode == "pertensor") {
        at::SmallVector<int64_t, op_infer::SIZE> per_tensor_size = {1};
        scale = npu_preparation::apply_tensor_without_format(per_tensor_size, c10::dtype(c10::ScalarType::Float));
        offset = npu_preparation::apply_tensor_without_format(per_tensor_size, c10::dtype(c10::ScalarType::Float));
    }

    dynamic_quant_run_aclnn<at::Tensor>(input, smooth_scales, group_index, y_wrapper, scale, offset, attr);

    return std::make_tuple(output, scale, offset);
}
}