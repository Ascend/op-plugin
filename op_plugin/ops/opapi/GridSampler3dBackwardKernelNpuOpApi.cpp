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

#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V1R11, V1R11)
std::tuple<at::Tensor, at::Tensor> grid_sampler_3d_backward(const at::Tensor& grad, const at::Tensor& input,
                                                            const at::Tensor& grid, int64_t interpolation_mode,
                                                            int64_t padding_mode, bool align_corners)
{
    DO_COMPATIBILITY(aclnnGridSampler3DBackward, acl_op::grid_sampler_3d_backward(grad, input, grid, interpolation_mode,
                                                                                  padding_mode, align_corners));
    at::Tensor dinput = npu_preparation::apply_tensor_without_format(input);
    at::Tensor dgrid = npu_preparation::apply_tensor_without_format(grid);
    std::array<bool, 2> output_mask = {true, true};
    EXEC_NPU_CMD(aclnnGridSampler3DBackward, grad, input, grid, interpolation_mode, padding_mode, align_corners,
                 output_mask, dinput, dgrid);
    return std::tuple<at::Tensor, at::Tensor>(dinput, dgrid);
}
#endif

}
