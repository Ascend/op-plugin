// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
using namespace op_infer;
namespace {
constexpr int64_t INT4_IN_INT32_NUM = 8;
constexpr int64_t MIN_INPUT_DIM = 2;
constexpr int64_t MAX_INPUT_DIM = 8;
constexpr int64_t DTYPE_NUM_FOR_QUINT4X2 = static_cast<int64_t>(at::ScalarType::QUInt4x2);
} // namespace

tensor_list npu_add_rms_norm_dynamic_quant(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma,
    const c10::optional<at::Tensor> &smooth_scale1, const c10::optional<at::Tensor> &smooth_scale2,
    const c10::optional<at::Tensor> &beta, double epsilon, const c10::optional<std::array<bool, 2>> output_mask,
    c10::optional<int64_t> y_dtype) {
    // 输出Tensor准备
    at::Tensor y1;
    at::Tensor y2;
    at::Tensor x_out;
    at::Tensor scale1;
    at::Tensor scale2;

    // 输出设置
    bool is_out_y1 = true;
    bool is_out_y2 = false;
    auto mask = c10::value_or_else(output_mask, [] {
        return std::array<bool, 2>{};
    });
    if (output_mask.has_value()) {
        TORCH_CHECK(!(!mask[0] && !mask[1]),
            "When the output_mask is not empty, at least one of y1 and y2 must be output. " +
                OPS_ERROR(ErrCode::PARAM));
        is_out_y1 = mask[0];
        is_out_y2 = mask[1];
    } else {
        is_out_y2 = smooth_scale1.has_value() && smooth_scale2.has_value();
    }

    // 参数检查
    TORCH_CHECK(
        x1.dim() >= MIN_INPUT_DIM && x1.dim() <= MAX_INPUT_DIM, "The x1 should be in 2~8D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        x2.dim() >= MIN_INPUT_DIM && x2.dim() <= MAX_INPUT_DIM, "The x2 should be in 2~8D" + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1.sizes() == x2.sizes(), "The shape of x1 and x2 must be the same" + OPS_ERROR(ErrCode::PARAM));

    // shape的初始化
    auto y_shape = array_to_small_vector(x1.sizes());
    c10::SmallVector<int64_t, op_infer::SIZE> scale_size;
    for (int index = 0; index < x1.dim() - 1; ++index) {
        scale_size.push_back(x1.size(index));
    }

    // y shape 和 dtype 推导以及 scale 推导
    aclDataType y_acltype;
    at::ScalarType scalar_dtype;
    bool special_output_type = false;
    if (y_dtype.has_value()) {
        special_output_type =
            (y_dtype == DTYPE_NUM_FOR_QUINT4X2 || c10_npu::GetAclDataType(y_dtype.value()) == aclDataType::ACL_INT4);
    }
    ASCEND_LOGI("[npu_add_rms_norm_dynamic_quant]: Getting aclTensor y dtype by Parameter(y_dtype): %ld",
        y_dtype.has_value() ? y_dtype.value() : -1L);
    if (special_output_type) {
        int64_t y_last_dim_val = y_shape[x1.dim() - 1];
        TORCH_CHECK(y_last_dim_val % INT4_IN_INT32_NUM == 0,
            "The last dim input shape must be divisible by 8 if y dtype is torch_npu.int4 or torch.quint4x2" +
                OPS_ERROR(ErrCode::PARAM));
        y_shape[x1.dim() - 1] = y_last_dim_val / INT4_IN_INT32_NUM;
        y_acltype = aclDataType::ACL_INT32;
        scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    } else {
        int64_t dst_type = c10::value_or_else(y_dtype, [] {
            return 1;
        });
        y_acltype = c10_npu::GetAclDataType(dst_type);
        scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
    }

    if (is_out_y1) {
        y1 = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
        scale1 = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(at::ScalarType::Float));
    } else {
        y1 = npu_preparation::apply_tensor_without_format(
            c10::SmallVector<int64_t, op_infer::SIZE>{}, c10::dtype(scalar_dtype));
        scale1 = npu_preparation::apply_tensor_without_format(
            c10::SmallVector<int64_t, op_infer::SIZE>{}, c10::dtype(at::ScalarType::Float));
    }
    if (is_out_y2) {
        y2 = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
        scale2 = npu_preparation::apply_tensor_without_format(scale_size, c10::dtype(at::ScalarType::Float));
    } else {
        y2 = npu_preparation::apply_tensor_without_format(
            c10::SmallVector<int64_t, op_infer::SIZE>{}, c10::dtype(scalar_dtype));
        scale2 = npu_preparation::apply_tensor_without_format(
            c10::SmallVector<int64_t, op_infer::SIZE>{}, c10::dtype(at::ScalarType::Float));
    }

    // x_out shape 推导
    auto x_out_shape = x1.sizes();
    auto x_out_dtype = x1.scalar_type();
    x_out = npu_preparation::apply_tensor_without_format(x_out_shape, c10::dtype(x_out_dtype));

    // 调用NPU原生算子执行
    TensorWrapper y1_wrapper = {y1, y_acltype};
    TensorWrapper y2_wrapper = {y2, y_acltype};

    if (output_mask.has_value()) {
        EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantV2, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, mask,
            y1_wrapper, y2_wrapper, x_out, scale1, scale2);
    } else {
        std::nullptr_t dummy_null = nullptr;
        EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantV2, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon,
            dummy_null, y1_wrapper, y2_wrapper, x_out, scale1, scale2);
    }

    return std::make_tuple(y1, y2, x_out, scale1, scale2);
}
} // namespace op_api
