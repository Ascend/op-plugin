// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "op_plugin/utils/OpAdapter.h"
#if VERSION_BETWEEN(V1R11, V2R1)
#include "op_plugin/AclOpsInterface.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {

at::Tensor& quantize_per_tensor_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype)
{
    string dtype_str = "torch.qint8";
    if (dtype == at::ScalarType::QUInt8) {
        dtype_str = "torch.quint8";
    } else if (dtype == at::ScalarType::QInt32) {
        dtype_str = "torch.qint32";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("Quantize")
       .Input(self)
       .Input(scales)
       .Input(zero_points)
       .Output(result)
       .Attr("axis", static_cast<int64_t>(1))
       .Attr("dtype", dtype_str)
       .Run();

    return result;
}
} // namespace

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
    quantize_per_tensor_out_nocheck(result, self, scale_tensor, zp_tensor, dtype);

    return result;
}
} // namespace acl_op
#endif

#if VERSION_BETWEEN(V2R2, VERSION_NEWEST)
#include "op_plugin/AclOpsInterface.h"
#include <ATen/ops/quantize_per_tensor.h>
#include <ATen/native/quantized/AffineQuantizer.h>

#include <torch/library.h>

namespace acl_op {
namespace {
at::Tensor& quantize_per_tensor_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    at::ScalarType dtype)
{
    string dtype_str = "torch.qint8";
    if (dtype == at::ScalarType::QUInt8) {
        dtype_str = "torch.quint8";
    } else if (dtype == at::ScalarType::QInt32) {
        dtype_str = "torch.qint32";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("Quantize")
       .Input(self)
       .Input(scales)
       .Input(zero_points)
       .Output(result)
       .Attr("axis", (int64_t)1)
       .Attr("dtype", dtype_str)
       .Run();

    return result;
}
} // namespace

at::Tensor quantize_per_tensor(
    const at::Tensor& self,
    double scale,
    int64_t zero_point,
    at::ScalarType dtype)
{
    return at::native::quantize_per_tensor(self, scale, zero_point, dtype);
}
} // namespace acl_op

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
    acl_op::quantize_per_tensor_out_nocheck(result, rtensor, scale_tensor, zero_point_tensor, dtype);
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
