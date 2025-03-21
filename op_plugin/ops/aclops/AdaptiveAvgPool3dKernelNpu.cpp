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
inline void adaptive_avg_pooling3d_check(const at::Tensor& self)
{
    for (int64_t i = 0; i < self.dim(); i++) {
        TORCH_CHECK(
            self.size(i) > 0,
            "adaptive_avg_pooling3d(): expected input to have non-empty spatial dimensions, "
            "but input has sizes ",
            self.sizes(),
            " with dimension ",
            i,
            " being "
            "empty" + OPS_ERROR(ErrCode::PARAM));
    }
    TORCH_CHECK(
        (self.dim() == 4 || self.dim() == 5),
        "non-empty 4D or 5D (batch mode) tensor expected for input" + OPS_ERROR(ErrCode::PARAM));
}

at::Tensor& adaptive_avg_pool3d_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef output_size)
{
    // reuse the mean out when d,h,w=1
    TORCH_CHECK(output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1,
        "adaptive_avg_pool3d only support D=1 && H=1 && W=1 current!" + OPS_ERROR(ErrCode::PARAM));
    at::mean_out(result, self, {self.dim() - 3, self.dim() - 2, self.dim() - 1}, true);

    return result;
}
} // namespace

at::Tensor& adaptive_avg_pool3d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& result)
{
    adaptive_avg_pooling3d_check(self);
    auto op_infer_output_size = op_infer::adaptive_avg_pool3d_npu_output_size(self, output_size);
    npu_preparation::CheckOut(
        {self},
        result,
        self,
        op_infer_output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        adaptive_avg_pool3d_out_nocheck(contiguous_result, self, output_size);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        adaptive_avg_pool3d_out_nocheck(result, self, output_size);
    }

    return result;
}

at::Tensor _adaptive_avg_pool3d(const at::Tensor& self, at::IntArrayRef output_size)
{
    adaptive_avg_pooling3d_check(self);
    auto op_infer_output_size = op_infer::adaptive_avg_pool3d_npu_output_size(self, output_size);
    at::Tensor result = npu_preparation::apply_tensor(self, op_infer_output_size);

    TORCH_CHECK(output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1,
        "adaptive_avg_pool3d only support D=1 && H=1 && W=1 current!" + OPS_ERROR(ErrCode::PARAM));
    return at::mean(self, {self.dim() - 3, self.dim() - 2, self.dim() - 1}, true);
}
} // namespace acl_op
