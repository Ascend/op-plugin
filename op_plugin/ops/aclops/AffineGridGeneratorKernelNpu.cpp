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
at::Tensor& affine_grid_generator_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& theta,
    at::IntArrayRef size,
    bool align_corners)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("AffineGrid")
        .Input(theta)
        .Input(size, at::kInt)
        .Output(result)
        .Attr("align_corners", align_corners)
        .Run();
    return result;
}
} // namespace

at::Tensor affine_grid_generator(
    const at::Tensor& theta,
    at::IntArrayRef size,
    bool align_corners)
{
    TORCH_CHECK(size.size() == 4 || size.size() == 5,
                "AffineGridGenerator needs 4d or 5d size(input)."
                + OPS_ERROR(ErrCode::PARAM));
    auto output_size = op_infer::infersize_affine_grid_generator(size);
    at::Tensor result = npu_preparation::apply_tensor(theta, output_size);
    affine_grid_generator_npu_nocheck(
        result,
        theta,
        size,
        align_corners);

    if (size.size() == 4) {
        result = result.view({size[0], size[2], size[3], 2});
    } else {
        result = result.view({size[0], size[2], size[3], size[4], 3});
    }
    return result;
}
} // namespace acl_op
