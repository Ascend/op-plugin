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

at::Tensor decode_jpeg(
    const at::Tensor& self,
    at::IntArrayRef image_shape,
    int64_t channels,
    bool try_recover_truncated)
{
    auto output_size = op_infer::decode_jpeg_npu_output_size(image_shape, channels);
    at::Tensor result = npu_preparation::apply_tensor_with_format(
        output_size,
        self.options().dtype(at::kByte),
        ACL_FORMAT_ND);

    at_npu::native::OpCommand cmd;
    cmd.Name("DecodeJpeg")
        .Input(self, "", c10::nullopt, "string")
        .Output(result)
        .Attr("channels", channels)
        .Attr("ratio", static_cast<int64_t>(1))
        .Attr("fancy_upscaling", static_cast<bool>(true))
        .Attr("try_recover_truncated", try_recover_truncated)
        .Attr("acceptable_fraction", static_cast<float>(1.0))
        .Attr("dct_method", static_cast<string>(""))
        .Attr("dst_img_format", static_cast<string>("CHW"))
        .Run();

    return result;
}
} // namespace acl_op
