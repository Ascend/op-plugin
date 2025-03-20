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
inline void max_unpool3d_check(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size)
{
    TORCH_CHECK(output_size.size() == 3, "There should be exactly 3 elements (depth, height, width) in output_size"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK((self.ndimension() == 4 || self.ndimension() == 5),
        "Input to max_unpooling2d should be a 4d or 5d Tensor"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.sizes() == indices.sizes(), "Shape of indices should match shape of input"
        + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.numel() > 0, "Input must be non-empty"
        + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor &max_unpool3d_out_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &indices,
                                     const at::Tensor &data)
{
    int64_t N = 1;
    int64_t C = self.size(0);
    if (self.dim() == 5) {
        N = self.size(0);
        C = self.size(1);
    }
    at::Tensor reshape_self = self.reshape({N, C, -1});
    at::Tensor reshape_indices = indices.reshape({N, C, -1});
    at::Tensor reshape_data = data.reshape({N, C, -1});
    result = result.reshape({N, C, -1});

    int64_t axis = 2;
    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterElements")
        .Input(reshape_data)
        .Input(reshape_indices)
        .Input(reshape_self)
        .Output(result)
        .Attr("axis", axis)
        .Run();
    result = result.reshape({data.sizes()});
    return result;
}
} // namespace

at::Tensor &max_unpool3d_out(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size,
                             at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor &out)
{
    max_unpool3d_check(self, indices, output_size);
    auto out_shape = op_infer::max_pool3d_output_size(self, output_size);
    at::Tensor data = at::zeros(out_shape, self.options());

    npu_preparation::CheckOut({self, indices, data}, out, data);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        max_unpool3d_out_nocheck(contiguous_result, self, indices, data);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        max_unpool3d_out_nocheck(out, self, indices, data);
    }

    return out;
}

at::Tensor max_unpool3d(const at::Tensor &self, const at::Tensor &indices, at::IntArrayRef output_size,
                        at::IntArrayRef stride, at::IntArrayRef padding)
{
    max_unpool3d_check(self, indices, output_size);
    auto out_shape = op_infer::max_pool3d_output_size(self, output_size);
    at::Tensor data = at::zeros(out_shape, self.options());
    at::Tensor result = npu_preparation::apply_tensor(data);
    max_unpool3d_out_nocheck(result, self, indices, data);

    return result;
}
} // namespace acl_op
