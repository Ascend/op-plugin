// Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
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

at::Tensor _cdist_backward(
    const at::Tensor& grad,
    const at::Tensor& x1,
    const at::Tensor& x2,
    double p,
    const at::Tensor& cdist)
{
    DO_COMPATIBILITY(aclnnCdistBackward, acl_op::_cdist_backward(grad, x1, x2, p, cdist));

    float p_cast;
    
    if (std::isinf(p)) {
        p_cast = -1;
    } else {
        TORCH_CHECK(
            p <= std::numeric_limits<float>::max(),
            "The value of p (" + std::to_string(p) + ") exceeds the maximum value of float ("
                + std::to_string(std::numeric_limits<float>::max()) + ")" + OPS_ERROR(ErrCode::PARAM));
        p_cast = static_cast<float>(p);
    }
    // The current operator has precision issues when handling integers and infinity.
    bool p_in_range = (p_cast >= 0.0 && p_cast <= 2.0) || (p_cast == -1);
    if (p_in_range) {
        return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
    }
    auto output_size = x1.sizes();
    auto output_dtype = grad.scalar_type();

    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        grad.options().dtype(output_dtype));

    EXEC_NPU_CMD(aclnnCdistBackward, grad, x1, x2, cdist, p_cast, out);

    return out;
}
}