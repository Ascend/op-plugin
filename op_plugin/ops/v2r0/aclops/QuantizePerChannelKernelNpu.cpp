// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
