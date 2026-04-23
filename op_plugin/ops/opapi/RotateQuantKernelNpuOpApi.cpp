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
constexpr int64_t DTYPE_NUM_FOR_QUINT4X2 = static_cast<int64_t>(at::ScalarType::QUInt4x2);
const int64_t INT4_IN_INT32_NUM = 8;

TensorWrapper get_output_tensor_wrapper(
    const at::Tensor &input, at::Tensor &output,
    aclDataType &y_acltype, c10::optional<int64_t> dst_type)
{
    auto x_size = input.sizes();
    if (dst_type == DTYPE_NUM_FOR_QUINT4X2) { // INT4
        TORCH_CHECK(input.size(1) % INT4_IN_INT32_NUM == 0,
                    "Input shape last dim must be divided by 8 when int4 quantization" + OPS_ERROR(ErrCode::PARAM));
        at::SmallVector<int64_t, op_infer::SIZE> input_shape_copy(input.sizes());
        input_shape_copy[1] /= INT4_IN_INT32_NUM;
        output = npu_preparation::apply_tensor_without_format(input_shape_copy, c10::dtype(c10::ScalarType::Int));
        y_acltype = aclDataType::ACL_INT32;
    } else { // 默认INT8
        output = npu_preparation::apply_tensor_without_format(x_size, c10::dtype(c10::ScalarType::Char));
        y_acltype = aclDataType::ACL_INT8;
    }
    TensorWrapper y_wrapper = {output, y_acltype};
    return y_wrapper;
}

std::tuple<at::Tensor, at::Tensor> npu_rotate_quant(
    const at::Tensor &x,
    const at::Tensor &rotation,
    c10::optional<double> alpha,
    c10::optional<int64_t> dst_dtype)
{
    auto x_size = x.sizes();
    int m = x_size[0];
    int n = x_size[1];
    at::Tensor output_y;
    aclDataType y_acltype;
    TensorWrapper y_wrapper = get_output_tensor_wrapper(x, output_y, y_acltype, dst_dtype);
    float alpha_real = static_cast<float>(alpha.value());

    at::Tensor output_scale = npu_preparation::apply_tensor_without_format({m}, c10::dtype( c10::ScalarType::Float));

    EXEC_NPU_CMD(
        aclnnRotateQuant,
        x,
        rotation,
        alpha_real,
        y_wrapper,
        output_scale);
    return std::tuple<at::Tensor, at::Tensor>(output_y, output_scale);
}
}