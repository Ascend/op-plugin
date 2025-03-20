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
at::Tensor _linspace_from_neg_one(const at::Tensor& grid, int64_t num_steps, bool align_corners)
{
    if (num_steps <= 1) {
        return at::tensor(0, grid.options());
    }
    auto range = at::linspace(-1, 1, num_steps, grid.options());
    if (!align_corners && num_steps != 0) {
        range = range * (num_steps - 1) / num_steps;
    }
    return range;
}

at::Tensor& affine_grid_generator_backward_nocheck(
    at::Tensor& result,
    const at::Tensor& grad,
    at::IntArrayRef size,
    bool align_corners)
{
    c10::SmallVector<int64_t, SIZE> output_size = {size[0], size[2], size[3], 3};
    at::Tensor assist = npu_preparation::apply_tensor(grad, output_size);
    assist.select(-1, 0).copy_(_linspace_from_neg_one(grad, size[3], align_corners));
    assist.select(-1, 1).copy_(_linspace_from_neg_one(grad, size[2], align_corners).unsqueeze_(-1));
    assist.select(-1, 2).fill_(1);
    AT_ASSERT(grad.sizes() == at::IntArrayRef({size[0], size[2], size[3], 2}), OPS_ERROR(ErrCode::VALUE));

    auto reassist = assist.view({size[0], size[2] * size[3], 3}).transpose(1, 2);
    auto grid = grad.view({size[0], size[2] * size[3], 2});

    at_npu::native::OpCommand cmd;
    cmd.Name("BatchMatMul")
        .Input(reassist)
        .Input(grid)
        .Output(result)
        .Attr("bias", (int64_t)0)
        .Attr("adj_x1", (bool)false)
        .Attr("adj_x2", (bool)false)
        .Run();

    return result;
}
} // namespace

at::Tensor affine_grid_generator_backward(
    const at::Tensor& grad,
    at::IntArrayRef size,
    bool align_corners)
{
    TORCH_CHECK(size.size() == 4, "AffineGridGeneratorBackward needs 4d (spatial) input."
        + OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, SIZE> output_size = {size[0], 3, 2};
    at::Tensor result = npu_preparation::apply_tensor_with_format(grad, output_size, ACL_FORMAT_ND);

    affine_grid_generator_backward_nocheck(
        result,
        grad,
        size,
        align_corners);
    auto fresult = result.transpose(1, 2);
    return fresult;
}
} // namespace acl_op
