// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

#if VERSION_BETWEEN(V2R1, V2R1)

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& _quantize_per_tensor_impl_out(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype,
    at::Tensor& result)
{
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    int64_t axis = 1;
    EXEC_NPU_CMD(aclnnQuantize, self, scales, zero_points, output_dtype, axis, result);

    return result;
}

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype)
{
    float scale_float = static_cast<float>(scale);
    auto output_size = op_infer::input_same_output_size(self);
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    at::Tensor scale_tensor = npu_preparation::apply_tensor_with_format(
        {1},
        self.options().dtype(at::kFloat),
        npu_preparation::get_tensor_npu_format(self));
    scale_tensor[0] = scale_float;
    at::Tensor zp_tensor = npu_preparation::apply_tensor_with_format(
        {1},
        self.options().dtype(at::kInt),
        npu_preparation::get_tensor_npu_format(self));
    zp_tensor[0] = zero_point;
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size,
        self.options().dtype(output_dtype),
        npu_preparation::get_tensor_npu_format(self));
    op_api::_quantize_per_tensor_impl_out(self, scale_tensor, zp_tensor, dtype, result);

    return result;
}
} // namespace op_api
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
#include "op_plugin/AclOpsInterface.h"
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/native/quantized/AffineQuantizer.h>

#include <torch/library.h>

namespace op_api {
at::Tensor& _quantize_per_tensor_impl_out(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype,
    at::Tensor& result)
{
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    int64_t axis = 1;
    EXEC_NPU_CMD(aclnnQuantize, self, scales, zero_points, output_dtype, axis, result);

    return result;
}

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype)
{
    return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}
} // namespace op_api

namespace at {
namespace native {
using npu_preparation = at_npu::native::OpPreparation;

void quantize_tensor_per_tensor_affine_npu(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    double scale,
    int64_t zero_point)
{
    float scale_float = static_cast<float>(scale);
    auto dtype = qtensor.options().dtype().toScalarType();
    auto output_size = op_infer::input_same_output_size(rtensor);
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    at::Tensor scale_tensor = npu_preparation::apply_tensor_with_format(
        {1},
        rtensor.options().dtype(at::kFloat),
        npu_preparation::get_tensor_npu_format(rtensor));
    scale_tensor[0] = scale_float;
    at::Tensor zero_point_tensor = npu_preparation::apply_tensor_with_format(
        {1},
        rtensor.options().dtype(at::kInt),
        npu_preparation::get_tensor_npu_format(rtensor));
    zero_point_tensor[0] = zero_point;
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size,
        rtensor.options().dtype(output_dtype),
        npu_preparation::get_tensor_npu_format(rtensor));
    at_npu::native::custom_ops::_quantize_per_tensor_impl_out(rtensor, scale_tensor,
        zero_point_tensor, dtype, result);
    at_npu::native::NPUNativeFunctions::set_(qtensor, result);
}

void dequantize_tensor_per_tensor_affine_npu(
    const at::Tensor& qtensor,
    at::Tensor& rtensor,
    double scale,
    int64_t zero_point)
{
    auto dtype = qtensor.scalar_type();
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    }
    at::Tensor result = at::empty(
        qtensor.sizes(),
        qtensor.options().dtype(output_dtype).memory_format(qtensor.suggest_memory_format()));

    at_npu::native::NPUNativeFunctions::set_(result, qtensor);
    rtensor = at_npu::native::custom_ops::npu_dtype_cast(result, at::ScalarType::Float);
    rtensor = (rtensor - zero_point) * scale;
}

REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_tensor_affine_stub, &quantize_tensor_per_tensor_affine_npu);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_tensor_affine_stub, &dequantize_tensor_per_tensor_affine_npu);

} // namespace native
} // namespace at
#endif
