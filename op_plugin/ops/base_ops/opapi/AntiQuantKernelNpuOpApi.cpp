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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor apply_anti_quant_out_tensor(const at::Tensor &x, at::ScalarType dst_type)
{
    if (x.dtype() == at::ScalarType::Int) {
        auto x_shape = op_infer::array_to_small_vector(x.sizes());
        size_t dim_num = x_shape.size();
        if (dim_num == 0) {
            TORCH_CHECK(false, "No supported for x is scalar when x dtype is int32 " + OPS_ERROR(ErrCode::TYPE));
        }

        x_shape[dim_num - 1] = x_shape[dim_num - 1] * 8;
        return at_npu::native::OpPreparation::apply_tensor(x_shape, x.options().dtype(dst_type), x);
    }

    return at_npu::native::OpPreparation::apply_tensor(x, x.options().dtype(dst_type));
}

at::Tensor npu_anti_quant(const at::Tensor &x, const at::Tensor &scale, const c10::optional<at::Tensor> &offset,
                          c10::optional<at::ScalarType> dst_dtype, c10::optional<at::ScalarType> src_dtype)
{
    auto input_dtype = x.dtype();
    if (input_dtype == at::ScalarType::Char) {
        if (src_dtype.has_value()) {
            TORCH_CHECK(src_dtype.value() == at::ScalarType::Char,
                        "x datatype is Int8, src_dtype must be same to x " + OPS_ERROR(ErrCode::TYPE));
        }
    } else if (input_dtype == at::ScalarType::Int) {
        if (src_dtype.has_value()) {
            TORCH_CHECK(src_dtype.value() == at::ScalarType::QUInt4x2,
                        "x datatype is Int32, src_dtype must be Int4 " + OPS_ERROR(ErrCode::TYPE));
        }
    } else {
        TORCH_CHECK(false, "Input x must be Int8 or Int32 " + OPS_ERROR(ErrCode::TYPE));
    }

    at::ScalarType dst_type = c10::value_or_else(dst_dtype, [] {return at::ScalarType::Half;});

    // construct the output tensor of the NPU
    at::Tensor result = apply_anti_quant_out_tensor(x, dst_type);

    bool sqrt_mode = false;

    if (src_dtype.has_value() && src_dtype.value() != input_dtype) {
        at::Tensor converted = x.to(src_dtype.value());
        EXEC_NPU_CMD(aclnnAscendAntiQuant, converted, scale, offset, dst_type, sqrt_mode, result);
    } else {
        EXEC_NPU_CMD(aclnnAscendAntiQuant, x, scale, offset, dst_type, sqrt_mode, result);
    }

    return result;
}
} // namespace op_api
