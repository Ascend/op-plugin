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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

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
