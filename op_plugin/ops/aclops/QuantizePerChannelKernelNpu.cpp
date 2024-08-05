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
c10::SmallVector<int64_t, SIZE> quantize_reshape_size(
    const at::Tensor& self,
    int64_t axis)
{
    c10::SmallVector<int64_t, SIZE> out_size;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i != axis) {
            out_size.emplace_back(1);
        } else {
            out_size.emplace_back(self.sizes()[i]);
        }
    }
    return out_size;
}

at::Tensor& quantize_per_channel_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    auto reshape_size = quantize_reshape_size(self, axis);
    at::Tensor scales_reshape = scales.reshape(reshape_size);
    at::Tensor zp_reshape = zero_points.reshape(reshape_size);
    string dtype_str = "torch.qint8";
    if (dtype == at::ScalarType::QUInt8) {
        dtype_str = "torch.quint8";
    } else if (dtype == at::ScalarType::QInt32) {
        dtype_str = "torch.qint32";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("Quantize")
       .Input(self)
       .Input(scales_reshape)
       .Input(zp_reshape)
       .Output(result)
       .Attr("axis", axis)
       .Attr("dtype", dtype_str)
       .Run();
    return result;
}
} // namespace

at::Tensor quantize_per_channel(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    axis = op_plugin::utils::make_warp_dim(axis, self.dim());
    TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
    TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
    TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
    TORCH_CHECK(axis <= self.sizes().size() - 1, "Unexpected value of axis.");
    TORCH_CHECK(scales.sizes()[0] == self.sizes()[axis], "length of scales must equal to the specified dimension.");
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    at::Tensor result = npu_preparation::apply_tensor(self, self.options().dtype(output_dtype));
    quantize_per_channel_out_nocheck(result, self, scales, zero_points, axis, dtype);
    return result;
}
} // namespace acl_op
#endif

#if VERSION_BETWEEN(V2R2, V2R2)
#include "op_plugin/AclOpsInterface.h"
#include <ATen/ops/quantize_per_channel.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <vector>

#include <torch/library.h>

namespace acl_op {
namespace {
c10::SmallVector<int64_t, SIZE> quantize_reshape_size(
    const at::Tensor& self,
    int64_t axis)
{
    c10::SmallVector<int64_t, SIZE> out_size;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i != axis) {
            out_size.emplace_back(1);
        } else {
            out_size.emplace_back(self.sizes()[i]);
        }
    }
    return out_size;
}

at::Tensor& quantize_per_channel_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    auto reshape_size = quantize_reshape_size(self, axis);
    at::Tensor scales_reshape = scales.reshape(reshape_size);
    at::Tensor zero_points_reshape = zero_points.reshape(reshape_size);
    string dtype_str = "torch.qint8";
    if (dtype == at::ScalarType::QUInt8) {
        dtype_str = "torch.quint8";
    } else if (dtype == at::ScalarType::QInt32) {
        dtype_str = "torch.qint32";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("Quantize")
       .Input(self)
       .Input(scales_reshape)
       .Input(zero_points_reshape)
       .Output(result)
       .Attr("axis", axis)
       .Attr("dtype", dtype_str)
       .Run();
    return result;
}
} // namespace

at::Tensor quantize_per_channel(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    auto zero_points_cpu = zero_points.to(at::Device(at::kCPU), at::kLong).contiguous();
    return at::native::quantize_per_channel(self, scales, zero_points_cpu, axis, dtype);
}
} // namespace acl_op

namespace at {
namespace native {
using npu_preparation = at_npu::native::OpPreparation;

void quantize_tensor_per_channel_affine_npu(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points_cpu,
    int64_t axis)
{
    auto zero_points = zero_points_cpu.to(at::Device(at::kPrivateUse1), at::ScalarType::Int).contiguous();
    at::ScalarType dtype = qtensor.options().dtype().toScalarType();
    axis = op_plugin::utils::make_warp_dim(axis, rtensor.dim());
    TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
    TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
    TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
    TORCH_CHECK(axis <= rtensor.sizes().size() - 1, "Unexpected value of axis.");
    TORCH_CHECK(scales.sizes()[0] == rtensor.sizes()[axis], "length of scales must equal to the specified dimension.");
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    at::Tensor result = npu_preparation::apply_tensor(rtensor, rtensor.options().dtype(output_dtype));
    acl_op::quantize_per_channel_out_nocheck(result, rtensor, scales, zero_points, axis, dtype);
    at_npu::native::NPUNativeFunctions::set_(qtensor, result);
}

void dequantize_tensor_per_channel_affine_npu(
    const at::Tensor& qtensor,
    at::Tensor& rtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points_cpu,
    int64_t axis)
{
    auto zero_points = zero_points_cpu.to(at::Device(at::kPrivateUse1), at::ScalarType::Int).contiguous();
    axis = op_plugin::utils::make_warp_dim(axis, qtensor.dim());
    TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
    TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
    TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
    TORCH_CHECK(axis <= qtensor.sizes().size() - 1, "Unexpected value of axis.");
    TORCH_CHECK(scales.sizes()[0] == qtensor.sizes()[axis], "length of scales must equal to the specified dimension.");

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

    auto reshape_size = acl_op::quantize_reshape_size(qtensor, axis);
    at::Tensor scales_reshape = scales.reshape(reshape_size);
    at::Tensor zero_points_reshape = zero_points.reshape(reshape_size);
    rtensor = (rtensor - zero_points_reshape) * scales_reshape;
}

REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_channel_affine_stub, &quantize_tensor_per_channel_affine_npu);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_channel_affine_stub, &dequantize_tensor_per_channel_affine_npu);

} // namespace native
} // namespace at
#endif

#if VERSION_BETWEEN(V2R3, VERSION_NEWEST)
#include "op_plugin/AclOpsInterface.h"
#include <ATen/ops/quantize_per_channel.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <vector>

#include <torch/library.h>

namespace acl_op {
namespace {
c10::SmallVector<int64_t, SIZE> quantize_reshape_size(
    const at::Tensor& self,
    int64_t axis)
{
    c10::SmallVector<int64_t, SIZE> out_size;
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i != axis) {
            out_size.emplace_back(1);
        } else {
            out_size.emplace_back(self.sizes()[i]);
        }
    }
    return out_size;
}

at::Tensor& quantize_per_channel_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    auto reshape_size = quantize_reshape_size(self, axis);
    at::Tensor scales_reshape = scales.reshape(reshape_size);
    at::Tensor zero_points_reshape = zero_points.reshape(reshape_size);
    string dtype_str = "torch.qint8";
    if (dtype == at::ScalarType::QUInt8) {
        dtype_str = "torch.quint8";
    } else if (dtype == at::ScalarType::QInt32) {
        dtype_str = "torch.qint32";
    }
    at_npu::native::OpCommand cmd;
    cmd.Name("Quantize")
       .Input(self)
       .Input(scales_reshape)
       .Input(zero_points_reshape)
       .Output(result)
       .Attr("axis", axis)
       .Attr("dtype", dtype_str)
       .Run();
    return result;
}
} // namespace

at::Tensor quantize_per_channel(
    const at::Tensor& self,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis,
    at::ScalarType dtype)
{
    return at::native::quantize_per_channel(self, scales, zero_points, axis, dtype);
}
} // namespace acl_op

namespace at {
namespace native {
using npu_preparation = at_npu::native::OpPreparation;

void quantize_tensor_per_channel_affine_npu(
    const at::Tensor& rtensor,
    at::Tensor& qtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis)
{
    at::ScalarType dtype = qtensor.options().dtype().toScalarType();
    axis = op_plugin::utils::make_warp_dim(axis, rtensor.dim());
    TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
    TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
    TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
    TORCH_CHECK(axis <= rtensor.sizes().size() - 1, "Unexpected value of axis.");
    TORCH_CHECK(scales.sizes()[0] == rtensor.sizes()[axis], "length of scales must equal to the specified dimension.");
    auto output_dtype = at::kInt;
    if (dtype == at::ScalarType::QInt8) {
        output_dtype = at::kChar;
    } else if (dtype == at::ScalarType::QUInt8) {
        output_dtype = at::kByte;
    } else if (dtype == at::ScalarType::QInt32) {
        output_dtype = at::kInt;
    }
    at::Tensor result = npu_preparation::apply_tensor(rtensor, rtensor.options().dtype(output_dtype));
    at::Tensor zero_points_cp = zero_points.to(at::ScalarType::Int).contiguous();
    acl_op::quantize_per_channel_out_nocheck(result, rtensor, scales, zero_points_cp, axis, dtype);
    at_npu::native::NPUNativeFunctions::set_(qtensor, result);
}

void dequantize_tensor_per_channel_affine_npu(
    const at::Tensor& qtensor,
    at::Tensor& rtensor,
    const at::Tensor& scales,
    const at::Tensor& zero_points,
    int64_t axis)
{
    axis = op_plugin::utils::make_warp_dim(axis, qtensor.dim());
    TORCH_CHECK(scales.dim() == 1, "Scales' dim should be equal to 1.");
    TORCH_CHECK(zero_points.dim() == 1, "Zero points' dim should be equal to 1.");
    TORCH_CHECK(scales.sizes()[0] == zero_points.sizes()[0], "Scales' size should be equal to zero points' size.");
    TORCH_CHECK(axis <= qtensor.sizes().size() - 1, "Unexpected value of axis.");
    TORCH_CHECK(scales.sizes()[0] == qtensor.sizes()[axis], "length of scales must equal to the specified dimension.");

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

    auto reshape_size = acl_op::quantize_reshape_size(qtensor, axis);
    at::Tensor scales_reshape = scales.reshape(reshape_size);
    at::Tensor zero_points_cp = zero_points.to(at::ScalarType::Int).contiguous();
    at::Tensor zero_points_reshape = zero_points_cp.reshape(reshape_size);
    rtensor = (rtensor - zero_points_reshape) * scales_reshape;
}

REGISTER_PRIVATEUSE1_DISPATCH(quantize_tensor_per_channel_affine_stub, &quantize_tensor_per_channel_affine_npu);
REGISTER_PRIVATEUSE1_DISPATCH(dequantize_tensor_per_channel_affine_stub, &dequantize_tensor_per_channel_affine_npu);

} // namespace native
} // namespace at
#endif