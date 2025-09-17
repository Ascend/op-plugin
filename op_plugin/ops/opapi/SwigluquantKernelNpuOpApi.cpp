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

namespace {
constexpr int64_t INT8 = 2;
constexpr int64_t INT4 = 29;
constexpr int64_t MAX_LAST_DIM = 8192;
// the last dimension of x should be divisible by 4
constexpr int64_t DIV_LAST_DIM = 4;
constexpr int64_t INT4_IN_INT8_NUM = 2;
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_swiglu_quant(const at::Tensor& x, const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& offsets, const c10::optional<at::Tensor>& group_index, bool activate_left,
    int64_t quant_mode, int64_t group_list_type, c10::optional<at::ScalarType> dst_type)
{
    TORCH_CHECK(quant_mode == 0 || quant_mode == 1, "quant_mode only support 0(static) or 1(dynamic), but got ",
                quant_mode, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1,
                "group_list_type only support 0(cumsum) or 1(count), but got ",
                group_list_type, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(!dst_type.has_value() || dst_type == at::ScalarType::Char || dst_type == at::ScalarType::QUInt4x2,
                "dtype must be torch.int8 for int8 or torch.quint4x2 for int4" + OPS_ERROR(ErrCode::TYPE));
    if (!dst_type.has_value()) {
        // dst_type default is torch.int8
        dst_type = at::ScalarType::Char;
    }

    const at::Tensor& smooth_scales_opt = c10::value_or_else(smooth_scales, [] { return at::Tensor(); });
    const at::Tensor& offsets_opt = c10::value_or_else(offsets, [] { return at::Tensor(); });
    TORCH_CHECK(quant_mode != 0 || smooth_scales_opt.sizes() == offsets_opt.sizes(),
                "smooth_scales and offsets should have the same shape when quant_mode is 0",
                OPS_ERROR(ErrCode::PARAM));
    
    // check x last dim
    int64_t x_last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(x_last_dim <= MAX_LAST_DIM, "x last dim size should not be larger than ", MAX_LAST_DIM, ", but got ",
                x_last_dim, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x_last_dim % 2 == 0, "x last dim size should be even", OPS_ERROR(ErrCode::PARAM));
    // to concatenate two INT4 into one INT8, the last dimension of x should be divisible by 4
    TORCH_CHECK(dst_type != at::ScalarType::QUInt4x2 || x_last_dim % DIV_LAST_DIM == 0,
                "x shape last dim must be divded by 4 when dst_type is torch.quint4x2, but got ",
                x_last_dim, OPS_ERROR(ErrCode::PARAM));

    at::SmallVector<int64_t, op_infer::SIZE> y_size;
    at::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int i = 0; i < x.dim() - 1; i++) {
        y_size.push_back(x.size(i));
        scale_size.push_back(x.size(i));
    }
    auto last_dim = dst_type == at::ScalarType::Char ? x_last_dim / 2 : x_last_dim / 2 / INT4_IN_INT8_NUM;
    y_size.push_back(last_dim);

    // The dtype of y is INT8(char), change y_size for different quantization types
    at::Tensor y = npu_preparation::apply_tensor_without_format(y_size, c10::dtype(c10::ScalarType::Char));
    at::Tensor scale = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(c10::ScalarType::Float));

    std::string quant_mode_str = quant_mode == 0 ? "static" : "dynamic";
    char* quant_mode_ptr = const_cast<char*>(quant_mode_str.c_str());

    const at::Tensor& group_index_opt = c10::value_or_else(group_index, [] { return at::Tensor(); });
    int output_type = dst_type == at::ScalarType::Char ? INT8 : INT4;
    EXEC_NPU_CMD(aclnnSwiGluQuantV2, x, smooth_scales_opt, offsets_opt, group_index_opt, activate_left, quant_mode_ptr,
        group_list_type, output_type, y, scale);

    return std::tie(y, scale);
}
}  // namespace op_api
