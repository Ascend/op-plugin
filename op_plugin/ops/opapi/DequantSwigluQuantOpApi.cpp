// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor, at::Tensor> npu_dequant_swiglu_quant(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale, const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale, const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index, bool activate_left, int64_t quant_mode)
{
    TORCH_CHECK(x.dim() > 1, "x dim should larger than 1", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0 or 1, but got", quant_mode,
                OPS_ERROR(ErrCode::PARAM));

    TORCH_CHECK(x.size(x.dim() - 1) % 2 == 0, "x last dim should be even", OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    auto last_dim = x.size(x.dim() - 1) / 2;
    y_size.push_back(last_dim);

    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    std::string quant_mode_str = "static";
    if (quant_mode == 1) {
        quant_mode_str = "dynamic";
    }
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& quant_scale_opt = c10::value_or_else(quant_scale, [] { return at::Tensor(); });
    const at::Tensor& quant_offset_opt = c10::value_or_else(quant_offset, [] { return at::Tensor(); });
    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });

    const at::Tensor& weight_scale_opt = c10::value_or_else(weight_scale, [] { return at::Tensor(); });
    const at::Tensor& activate_scale_opt = c10::value_or_else(activation_scale, [] { return at::Tensor(); });
    const at::Tensor& bias_opt = c10::value_or_else(bias, [] { return at::Tensor(); });

    EXEC_NPU_CMD(aclnnDequantSwigluQuant, x, weight_scale_opt, activate_scale_opt, bias_opt, quant_scale_opt,
                 quant_offset_opt, group_index_opt, activate_left, quant_mode_ptr, y, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
