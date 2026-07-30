// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

constexpr int64_t TND_DIMS = 3;
constexpr int64_t BSND_DIMS = 4;
constexpr int64_t INDEX_TWO = 2;

inline void check_mhc_post_backward_supported() {
    static const bool is_cann_ready = op_plugin::utils::is_gte_cann_version_900();
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnMhcPostBackward");
    TORCH_CHECK(is_cann_ready && is_aclnn_kernel_available,
        "torch_npu.npu_mhc_post_backward requires CANN >= 9.0.0, aclnnMhcPostBackward support. "
        "Please upgrade CANN.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_mhc_post_backward(const at::Tensor &grad_y,
    const at::Tensor &x, const c10::optional<at::Tensor> &h_res, const at::Tensor &h_out, const at::Tensor &h_post) {
    TORCH_CHECK(grad_y.numel() > 0, "Tensor grad_y is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_out.numel() > 0, "Tensor h_out is empty.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_post.numel() > 0, "Tensor h_post is empty.", OPS_ERROR(ErrCode::PARAM));

    auto x_dtype = x.scalar_type();
    TORCH_CHECK(x_dtype == at::kHalf || x_dtype == at::kBFloat16, "x dtype must be FLOAT16 or BFLOAT16, but got ",
        x_dtype, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grad_y.scalar_type() == x_dtype, "grad_y dtype must be the same as x, but got ", grad_y.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_out.scalar_type() == x_dtype, "h_out dtype must be the same as x, but got ", h_out.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_post.scalar_type() == at::kFloat, "h_post dtype must be FLOAT32, but got ", h_post.scalar_type(),
        OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(x.dim() == TND_DIMS || x.dim() == BSND_DIMS, "x dim must be 3 or 4, but got ", x.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(grad_y.sizes() == x.sizes(), "grad_y shape must be the same as x shape.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_out.dim() == x.dim() - 1, "h_out dim must be ", x.dim() - 1, ", but got ", h_out.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(h_post.dim() == x.dim() - 1, "h_post dim must be ", x.dim() - 1, ", but got ", h_post.dim(),
        OPS_ERROR(ErrCode::PARAM));

    auto grad_h_res_size = x.sizes().vec();
    grad_h_res_size.back() = x.size(x.dim() - INDEX_TWO);

    at::Tensor h_res_tensor = h_res.value_or(at::Tensor());
    if (h_res_tensor.defined()) {
        TORCH_CHECK(h_res_tensor.numel() > 0, "Tensor h_res is empty.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(h_res_tensor.scalar_type() == at::kFloat, "h_res dtype must be FLOAT32, but got ",
            h_res_tensor.scalar_type(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(h_res_tensor.sizes() == c10::IntArrayRef(grad_h_res_size), "h_res shape must be ",
            c10::IntArrayRef(grad_h_res_size), ", but got ", h_res_tensor.sizes(), OPS_ERROR(ErrCode::PARAM));
    }

    check_mhc_post_backward_supported();

    at::Tensor grad_x = npu_preparation::apply_tensor_without_format(x);
    at::Tensor grad_h_res =
        npu_preparation::apply_tensor_without_format(grad_h_res_size, x.options().dtype(at::kFloat));
    at::Tensor grad_h_out = npu_preparation::apply_tensor_without_format(h_out);
    at::Tensor grad_h_post = npu_preparation::apply_tensor_without_format(h_post);

    EXEC_NPU_CMD(
        aclnnMhcPostBackward, grad_y, x, h_res_tensor, h_out, h_post, grad_x, grad_h_res, grad_h_out, grad_h_post);

    return std::make_tuple(grad_x, grad_h_res, grad_h_out, grad_h_post);
}

} // namespace op_api
