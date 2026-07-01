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
    if (p_in_range && (c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend950)) {
        return acl_op::_cdist_backward(grad, x1, x2, p, cdist);
    }

    int64_t c1 = x1.size(-1);
    int64_t c2 = x2.size(-1);
    int64_t r1 = x1.size(-2);
    int64_t r2 = x2.size(-2);
    int64_t dim1 = static_cast<int64_t>(x1.dim());
    int64_t dim2 = static_cast<int64_t>(x2.dim());
    TORCH_CHECK(c1 == c2, "X1 and X2 must have the same number of columns. X1: ", c1, " X2: ", c2,
        OPS_ERROR(ErrCode::PARAM));

    at::IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
    at::IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
    std::vector<int64_t> expand_batch_portion = at::infer_size(batch_tensor1, batch_tensor2);
    std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
    tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
    std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
    tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

    bool empty_batch = false;
    for (const auto size : expand_batch_portion) {
        empty_batch = empty_batch || (size == 0);
    }
    if (r1 == 0 || r2 == 0 || c1 == 0 || empty_batch) {
        return at::zeros_like(x1, x1.options());
    }

    at::Tensor x1_broadcast = x1;
    if (x1.sizes().vec() != tensor1_expand_size) {
        x1_broadcast = x1.expand(tensor1_expand_size);
    }
    x1_broadcast = x1_broadcast.contiguous();

    at::Tensor x2_broadcast = x2;
    if (x2.sizes().vec() != tensor2_expand_size) {
        x2_broadcast = x2.expand(tensor2_expand_size);
    }
    x2_broadcast = x2_broadcast.contiguous();

    auto grad_contiguous = grad.contiguous();
    auto cdist_contiguous = cdist.contiguous();
    auto output_size = x1_broadcast.sizes();
    auto output_dtype = x1_broadcast.scalar_type();
    at::Tensor out = at_npu::native::OpPreparation::apply_tensor_without_format(
        output_size,
        x1_broadcast.options().dtype(output_dtype));

    EXEC_NPU_CMD(aclnnCdistBackward, grad_contiguous, x1_broadcast, x2_broadcast, cdist_contiguous, p_cast, out);

    return out;
}
}
