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
const int64_t INT4_IN_INT32_NUM = 8;

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant_v0(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<at::ScalarType> dst_type)
{
    TORCH_CHECK(dst_type != at::ScalarType::QUInt4x2,
                "please update your CANN to support int4 quantization" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    TORCH_CHECK(!group_index.has_value(),
                "please update your CANN to support MOE quantization" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    for (int i = 0; i < scale_dim; ++i) {
        scale_size.push_back(input.size(i));
    }

    at::Tensor output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    EXEC_NPU_CMD(aclnnDynamicQuant, input, smooth_scales, output, scale);
    return std::make_tuple(output, scale);
}

std::tuple<at::Tensor, at::Tensor> npu_dynamic_quant(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<at::ScalarType> dst_type)
{
    TORCH_CHECK(!dst_type.has_value() || dst_type == at::ScalarType::Char || dst_type == at::ScalarType::QUInt4x2,
                "dtype must be torch.int8 for int8 or torch.quint4x2 for int4" + OPS_ERROR(ErrCode::TYPE));
    DO_COMPATIBILITY(aclnnDynamicQuantV2, npu_dynamic_quant_v0(input, smooth_scales, group_index, dst_type));
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    int output_type;
    at::Tensor output;
    if (dst_type == at::ScalarType::QUInt4x2) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "input shape last dim must be divded by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        output_type = ge::DataType::DT_INT32;
        scale_size.push_back(input.size(index) / INT4_IN_INT32_NUM);
        output = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Int));
    } else { // 默认INT8
        output_type = ge::DataType::DT_INT8;
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    }
    c10::optional<at::Tensor> offset;

    EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index, output_type, output, scale, offset);
    return std::make_tuple(output, scale);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dynamic_quant_asymmetric(
    const at::Tensor &input,
    const c10::optional<at::Tensor> &smooth_scales,
    const c10::optional<at::Tensor> &group_index,
    c10::optional<at::ScalarType> dst_type)
{
    TORCH_CHECK(!dst_type.has_value() || dst_type == at::ScalarType::Char || dst_type == at::ScalarType::QUInt4x2,
                "dtype must be torch.int8 for int8 or torch.quint4x2 for int4 when int4 quantization" + OPS_ERROR(ErrCode::TYPE));
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    int scale_dim = input.dim() - 1;
    int index = 0;
    for (; index < scale_dim; ++index) {
        scale_size.push_back(input.size(index));
    }

    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    at::Tensor offset = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));
    int output_type;
    at::Tensor output;
    if (dst_type == at::ScalarType::QUInt4x2) { // INT4
        TORCH_CHECK(input.size(index) % INT4_IN_INT32_NUM == 0,
                    "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_type = ge::DataType::DT_INT32;
        scale_size.push_back(input.size(index) / INT4_IN_INT32_NUM);
        output = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Int));
    } else { // 默认INT8
        output_type = ge::DataType::DT_INT8;
        output = npu_preparation::apply_tensor_without_format(input.sizes(), c10::dtype(c10::ScalarType::Char));
    }

    EXEC_NPU_CMD(aclnnDynamicQuantV2, input, smooth_scales, group_index, output_type, output, scale, offset);
    return std::make_tuple(output, scale, offset);
}
}