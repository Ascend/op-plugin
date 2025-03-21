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
using npu_utils = at_npu::native::NpuUtils;

namespace {
inline void max_unpool2d_check(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size)
{
    TORCH_CHECK(output_size.size() == 2, "There should be exactly two elements (height, width) in output_size"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 3 || self.ndimension() == 4),
        "Input to max_unpooling2d should be a 3d or 4d Tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.sizes() == indices.sizes(), "Shape of indices should match shape of input"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.numel() > 0, "Input must be non-empty"
        + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor &max_unpool2d_out_nocheck(at::Tensor &output, const at::Tensor &self, const at::Tensor &indices,
                                     at::IntArrayRef output_size)
{
    auto oheight = output_size[0];
    auto owidth = output_size[1];
    auto self_contiguous = self.contiguous();
    auto indices_contiguous = indices.contiguous();
    int64_t h = -1;
    int64_t w = -1;
    int64_t self_dim = self.ndimension();
    int64_t num_batch = -1;
    int64_t num_channels = -1;
    if (self_dim == 3) {
        num_channels = self.size(0);
        h = self.size(1);
        w = self.size(2);
        output.resize_({num_channels, oheight * owidth});
        self_contiguous = self_contiguous.reshape({num_channels, h * w});
        indices_contiguous = indices_contiguous.reshape({num_channels, h * w});
    } else {
        num_batch = self.size(0);
        num_channels = self.size(1);
        h = self.size(2);
        w = self.size(3);
        output.resize_({num_batch, num_channels, oheight * owidth});
        self_contiguous = self_contiguous.reshape({num_batch, num_channels, h * w});
        indices_contiguous = indices_contiguous.reshape({num_batch, num_channels, h * w});
    }

    output.zero_();
    int64_t dim = 2;
    output = output.scatter(dim, indices_contiguous, self_contiguous);
    if (self_dim == 3) {
        output = output.reshape({num_channels, oheight, owidth});
    } else {
        output = output.reshape({num_batch, num_channels, oheight, owidth});
    }
    return output;
}
} // namespace

at::Tensor &max_unpool2d_out(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size,
                             at::Tensor &out)
{
    max_unpool2d_check(self, indices, output_size);
    npu_preparation::CheckOut({self, indices}, out, self, {0});
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_output = npu_utils::format_contiguous(out);
        max_unpool2d_out_nocheck(contiguous_output, self, indices, output_size);
        npu_utils::format_fresh_view(out, contiguous_output);
    } else {
        max_unpool2d_out_nocheck(out, self, indices, output_size);
    }

    return out;
}

at::Tensor max_unpool2d(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size)
{
    max_unpool2d_check(self, indices, output_size);
    auto output = npu_preparation::apply_tensor(self, {0});
    max_unpool2d_out_nocheck(output, self, indices, output_size);
    return output;
}
} // namespace acl_op
