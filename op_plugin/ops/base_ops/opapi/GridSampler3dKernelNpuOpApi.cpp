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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor grid_sampler_3d(const at::Tensor& self, const at::Tensor& grid, int64_t interpolation_mode,
                           int64_t padding_mode, bool align_corners)
{
    DO_COMPATIBILITY(aclnnGridSampler3D, acl_op::grid_sampler_3d(self, grid, interpolation_mode,
                                                                 padding_mode, align_corners));
    auto output_size = {self.size(0), self.size(1), grid.size(1), grid.size(2), grid.size(3)};
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options());
    EXEC_NPU_CMD(aclnnGridSampler3D, self, grid, interpolation_mode, padding_mode, align_corners, result);
    return result;
}
} // namespace op_api
