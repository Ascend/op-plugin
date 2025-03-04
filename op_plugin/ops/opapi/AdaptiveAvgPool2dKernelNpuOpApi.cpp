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

at::Tensor& adaptive_avg_pool2d_out(const at::Tensor& self, at::IntArrayRef output_size, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, acl_op::adaptive_avg_pool2d_out(self, output_size, result));
    TORCH_CHECK((output_size.size() == 2), "adaptive_avg_pool2d: output size must be 2." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dtype() == result.dtype(),
                "expected dtype ", self.dtype(),
                " for `output` but got dtype ", result.dtype(), OPS_ERROR(ErrCode::TYPE));
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool2d, self, output_size, result);
    return result;
}

at::Tensor adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, acl_op::adaptive_avg_pool2d(self, output_size));
    // The logic is a little different from CPU_impl
    // can't use "NPUNativeOpApiFunctions::_adaptive_avg_pool2d(self, output_size)", this will resnet50 accuracy error
    TORCH_CHECK((output_size.size() == 2), "adaptive_avg_pool2d: output size must be 2." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        (output_size[0] >= 0 && output_size[1] >= 0),
        "adaptive_avg_pool2d: elements of output_size must be greater than or equal to 0 ",
        "but received {", output_size[0], ", ", output_size[1], "}" + OPS_ERROR(ErrCode::PARAM));
    if (!self.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
        // in this case, adaptive pooling is just computing mean over hw
        // dimensions, which can be done more efficiently
        at::Tensor result = self.mean({-1, -2}, true);
        return result;
    }
    return at::_adaptive_avg_pool2d(self, output_size);
}

at::Tensor _adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool2d, acl_op::_adaptive_avg_pool2d(self, output_size));
    TORCH_CHECK((self.dim() == 3 || self.dim() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input"
        + OPS_ERROR(ErrCode::PARAM));
    auto outputSize = op_infer::array_to_small_vector(self.sizes());
    outputSize[self.dim() - 1] = output_size[1];
    outputSize[self.dim() - 2] = output_size[0];
    at::Tensor result = npu_preparation::apply_tensor_without_format(self, outputSize);
    op_api::adaptive_avg_pool2d_out(self, output_size, result);
    return result;
}

}
