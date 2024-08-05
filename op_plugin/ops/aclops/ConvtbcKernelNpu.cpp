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

at::Tensor conv_tbc(const at::Tensor &self, const at::Tensor &weight, const at::Tensor &bias, int64_t pad)
{
    TORCH_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, in_channel." + OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
        " in_channels, out_channels." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1-D." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(2) == weight.size(1), "Input dim 2 (input channels) "
        "is not == dim 1 in the weight tenso." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.size(2) == bias.size(0), "Bias size must equal dim 2 in "
        "the weight tensor (output channels)." + OPS_ERROR(ErrCode::PARAM));

    int64_t C = weight.size(2);
    int64_t W = (self.size(0) + 2 * pad - (weight.size(0) - 1) - 1) + 1;
    TORCH_CHECK(W > 0, "W has to be positive, but got ", W, OPS_ERROR(ErrCode::VALUE));

    c10::SmallVector<int64_t, SIZE> output_size = {self.size(1), C, 1, W};

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_with_format(self, output_size, ACL_FORMAT_NCHW);

    c10::SmallVector<int64_t, N> paddings = {0, 0, pad, pad};
    c10::SmallVector<int64_t, N> strides_size = {1, 1, 1, 1};
    c10::SmallVector<int64_t, N> dilations = {1, 1, 1, 1};

    at::Tensor self_tensor = self.transpose(0, 2).transpose(0, 1).unsqueeze(2);
    at::Tensor weight_tensor = weight.transpose(0, 2).unsqueeze(2);

    at_npu::native::OpCommand cmd;
    cmd.Name("Conv2D")
        .Input(self_tensor, "x")
        .Input(weight_tensor, "filter")
        .Input(bias)
        .Output(result, "y")
        .Attr("pads", paddings)
        .Attr("strides", strides_size)
        .Attr("dilations", dilations)
        .Attr("data_format", (string) "NCHW")
        .Run();

    result = result.squeeze(2).transpose(0, 2).transpose(1, 2);
    return result;
}
} // namespace acl_op
