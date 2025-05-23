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
at::Tensor affine_grid_generator(const at::Tensor& theta, at::IntArrayRef size, bool align_corners)
{
    DO_COMPATIBILITY(aclnnAffineGrid, acl_op::affine_grid_generator(theta, size, align_corners));
    TORCH_CHECK(size.size() == 4 || size.size() == 5, "AffineGridGenerator needs 4d or 5d size(input)."
        + OPS_ERROR(ErrCode::PARAM));
    at::SmallVector<int64_t, SIZE> outputSize = {};
    if (size.size() == 4) {
        outputSize = {size[0], size[2], size[3], 2};
    } else {
        outputSize = {size[0], size[2], size[3], size[4], 3};
    }

    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_without_format(theta, outputSize);
    EXEC_NPU_CMD(aclnnAffineGrid, theta, size, align_corners, result);
    return result;
}
}
