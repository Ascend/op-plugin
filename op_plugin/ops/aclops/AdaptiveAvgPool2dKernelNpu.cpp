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
inline void adaptive_avg_pool2d_check(const at::Tensor& self)
{
    for (int64_t i = 0; i < self.dim(); i++) {
        TORCH_CHECK(
            self.size(i) > 0,
            "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
            "but input has sizes ",
            self.sizes(),
            " with dimension ",
            i,
            " being "
            "empty" + OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(
        (self.dim() == 3 || self.dim() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input" + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor& adaptive_avg_pool2d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size)
{
    if (output_size[0] == 1 && output_size[1] == 1) {
        at::mean_out(result, self, {self.dim() - 2, self.dim() - 1}, true);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("AdaptiveAvgPool2d")
            .Input(self)
            .Output(result)
            .Attr("output_size", output_size)
            .Run();
    }

    return result;
}
} // namespace

at::Tensor& adaptive_avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& out)
{
    adaptive_avg_pool2d_check(self);
    auto op_infer_output_size = op_infer::array_to_small_vector(self.sizes());
    op_infer_output_size[self.dim() - 1] = output_size[1];
    op_infer_output_size[self.dim() - 2] = output_size[0];
    npu_preparation::CheckOut(
        {self},
        out,
        npu_preparation::get_tensor_npu_format(out),
        self.scalar_type(),
        op_infer_output_size);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out);
        adaptive_avg_pool2d_out_nocheck(contiguous_out, self, output_size);
        npu_utils::format_fresh_view(out, contiguous_out);
    } else {
        adaptive_avg_pool2d_out_nocheck(out, self, output_size);
    }

    return out;
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size)
{
    // The logic is a little different from CPU_impl
    return at::_adaptive_avg_pool2d(self, output_size);
}

at::Tensor _adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size)
{
    adaptive_avg_pool2d_check(self);
    auto op_infer_output_size = op_infer::array_to_small_vector(self.sizes());
    op_infer_output_size[self.dim() - 1] = output_size[1];
    op_infer_output_size[self.dim() - 2] = output_size[0];

    at::Tensor result = npu_preparation::apply_tensor(self, op_infer_output_size);
    adaptive_avg_pool2d_out_nocheck(result, self, output_size);

    return result;
}
} // namespace acl_op
