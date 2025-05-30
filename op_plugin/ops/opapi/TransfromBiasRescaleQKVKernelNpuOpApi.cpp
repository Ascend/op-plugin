// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

inline bool _transform_bias_rescale_qkv_fallback_condition()
{
    static const bool is_aclnn_kernel_available = check_aclnn_kernel_available("aclnnTransformBiasRescaleQkv");
    if (!is_aclnn_kernel_available) {
        TORCH_NPU_WARN_ONCE("CAUTION: The operator aten::_transform_bias_rescale_qkv is currently "
            "not supported on the NPU backend. Now this operator will fallback to run on the CPU "
            "and may have performance implications. Please try to update your CANN version.");
        return true;
    }
    return false;
}

c10::SmallVector<int64_t, SIZE> _transform_bias_rescale_qkv_out_size(
    const at::Tensor& qkv,
    const at::Tensor& qkv_bias,
    const int64_t num_head)
{
    TORCH_CHECK(num_head != 0, "num_head of _transform_bias_rescale_qkv must not be 0.", OPS_ERROR(ErrCode::VALUE))
    auto B = qkv.size(0);
    auto T = qkv.size(1);
    auto _3D = qkv_bias.size(0);
    auto D = _3D / 3;
    TORCH_CHECK(D % num_head == 0, "D of _transform_bias_rescale_qkv must divide evenly by num_head", OPS_ERROR(ErrCode::VALUE));
    const auto dim_per_head = D / num_head;
    c10::SmallVector<int64_t, SIZE> output_shape = {B, num_head, T, dim_per_head};
    
    return output_shape;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> _transform_bias_rescale_qkv(
    const at::Tensor& qkv,
    const at::Tensor& qkv_bias,
    const int64_t num_heads)
{
    if (_transform_bias_rescale_qkv_fallback_condition()) {
        at::Tensor qkv_cpu = qkv.cpu();
        at::Tensor qkv_bias_cpu = qkv_bias.cpu();
        std::tuple<at::Tensor, at::Tensor, at::Tensor> result_cpu = at::_transform_bias_rescale_qkv(qkv_cpu, qkv_bias_cpu, num_heads);
        at::Tensor q = std::get<0>(result_cpu).to(qkv.device());
        at::Tensor k = std::get<1>(result_cpu).to(qkv.device());
        at::Tensor v = std::get<2>(result_cpu).to(qkv.device());
        return std::make_tuple(std::move(q), std::move(k), std::move(v));
    }

    // Currently, since NPU don't support Nested Tensor, related logic will not be added for now.
    auto output_size = _transform_bias_rescale_qkv_out_size(qkv, qkv_bias, num_heads);
    auto output_dtype = qkv.scalar_type();

    at::Tensor q = npu_preparation::apply_tensor_without_format(output_size, qkv.options().dtype(output_dtype));
    at::Tensor k = npu_preparation::apply_tensor_without_format(output_size, qkv.options().dtype(output_dtype));
    at::Tensor v = npu_preparation::apply_tensor_without_format(output_size, qkv.options().dtype(output_dtype));
    EXEC_NPU_CMD(aclnnTransformBiasRescaleQkv, qkv, qkv_bias, num_heads, q, k, v);
    return std::make_tuple(std::move(q), std::move(k), std::move(v));
}

}  // namespace op_api
