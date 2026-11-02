// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#if defined(__has_include)
#if __has_include("ATen/native/GroupedMMUtils.h")
#include "ATen/native/GroupedMMUtils.h"
#define HAS_GROUPED_MM_FALLBACK 1
#else
#define HAS_GROUPED_MM_FALLBACK 0
#endif
#else
#define HAS_GROUPED_MM_FALLBACK 0
#endif

namespace op_api {
const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
const static int64_t IN_SPLIT_OUT_SPLIT = 3;
const static int64_t M_SPLIT = 0;
const static int64_t K_SPLIT = 2;
const static int64_t ACT_TYPE_NONE = 0;
const static int64_t GROUP_LIST_TYPE_CUMSUM = 0;
using npu_preparation = at_npu::native::OpPreparation;

// Execute aclnnGroupedMatmulV5.
// All optional quantization/activation parameters are passed as empty TensorLists.
// This mirrors how npu_grouped_matmul calls V5 — pure data forwarding, no scenario logic.
// V5 availability is checked at the _grouped_mm entry point (with fallback to the native
// at::mm_out loop). The TORCH_CHECK below is defensive — it should never trigger because
// the entry point already handles the unavailable case before calling this function.
static void exec_grouped_matmul_v5(
    const std::vector<at::Tensor>& x_vec,
    const std::vector<at::Tensor>& weight_vec,
    const at::Tensor& group_list,
    int64_t split_item,
    int64_t group_type,
    std::vector<at::Tensor>& result_vec)
{
    auto x = at::TensorList(x_vec);
    auto weight = at::TensorList(weight_vec);
    auto bias_real = at::TensorList();
    auto scale_real = at::TensorList();
    auto offset_real = at::TensorList();
    auto antiquant_scale_real = at::TensorList();
    auto antiquant_offset_real = at::TensorList();
    auto per_token_scale_real = at::TensorList();
    auto activation_input_real = at::TensorList();
    auto activation_quant_scale_real = at::TensorList();
    auto activation_quant_offset_real = at::TensorList();
    auto act_out = at::TensorList();
    auto dynamic_quant_scale_out = at::TensorList();
    auto result = at::TensorList(result_vec);
    int64_t group_list_type_value = GROUP_LIST_TYPE_CUMSUM;
    int64_t act_type_value = ACT_TYPE_NONE;
    auto tuning_config_real = at::IntArrayRef{};

    static const bool is_v5_available = check_aclnn_kernel_available("aclnnGroupedMatmulV5");
    TORCH_CHECK(is_v5_available,
                "aclnnGroupedMatmulV5 is not available, please update CANN to a newer version.",
                OPS_ERROR(ErrCode::INTERNAL));
    EXEC_NPU_CMD(aclnnGroupedMatmulV5, x, weight, bias_real, scale_real, offset_real,
        antiquant_scale_real, antiquant_offset_real, per_token_scale_real, group_list,
        activation_input_real, activation_quant_scale_real, activation_quant_offset_real,
        split_item, group_type, group_list_type_value, act_type_value, tuning_config_real,
        result, act_out, dynamic_quant_scale_out);
}

// Convert offs from int32 to int64, ensure contiguous.
// The NPU API requires group_list to be int64.
static at::Tensor prepare_group_list(const at::Tensor& offs)
{
    at::Tensor group_list = offs.to(at::kLong);
    if (group_list.is_contiguous()) {
        return group_list;
    }
    return group_list.contiguous();
}

at::Tensor _grouped_mm(const at::Tensor& mat_a, const at::Tensor& mat_b,
                       const c10::optional<at::Tensor>& offs,
                       const c10::optional<at::Tensor>& bias,
                       c10::optional<c10::ScalarType> out_dtype)
{
    // Validate inputs (mirrors native PyTorch GroupedMMUtils.h checks)
    TORCH_CHECK(mat_a.dtype() == at::kBFloat16 || mat_a.dtype() == at::kFloat || mat_a.dtype() == at::kHalf,
                "Expected mat_a to be Float32, BFloat16 or Float16 matrix, got ", mat_a.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_b.dtype() == at::kBFloat16 || mat_b.dtype() == at::kFloat || mat_b.dtype() == at::kHalf,
                "Expected mat_b to be Float32, BFloat16 or Float16 matrix, got ", mat_b.scalar_type(),
                OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_a.dtype() == mat_b.dtype(),
                "Expected mat_a and mat_b to have the same dtype, got ",
                mat_a.scalar_type(), " and ", mat_b.scalar_type(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3,
                "mat_a has to be 2 or 3d", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3,
                "mat_b has to be 2 or 3d", OPS_ERROR(ErrCode::PARAM));

    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;

    // Check contraction dimension matches
    TORCH_CHECK(mat_a.size(-1) == mat_b.size(-2),
                "contraction dimension of mat_a and mat_b must match",
                OPS_ERROR(ErrCode::PARAM));

    // Check offs presence: required when any input is 2D, forbidden when both are 3D
    TORCH_CHECK(offs.has_value() == (a_is_2d || b_is_2d),
                "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d",
                OPS_ERROR(ErrCode::PARAM));

    if (offs.has_value()) {
        TORCH_CHECK(offs->dim() == 1, "offs has to be 1D", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32",
                    OPS_ERROR(ErrCode::TYPE));
    }

    // bias not supported yet (consistent with native PyTorch)
    TORCH_CHECK(!bias.has_value(), "Bias not supported yet", OPS_ERROR(ErrCode::PARAM));

    // Resolve output dtype
    const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
    TORCH_CHECK(out_dtype_ == mat_a.dtype(),
                "Grouped gemm output dtype must match mat_a dtype",
                OPS_ERROR(ErrCode::TYPE));

    auto options = mat_a.options().dtype(out_dtype_);

    // Fallback: if aclnnGroupedMatmulV5 is unavailable
    static const bool is_v5_available_fallback = check_aclnn_kernel_available("aclnnGroupedMatmulV5");
    if (!is_v5_available_fallback) {
#if HAS_GROUPED_MM_FALLBACK
        std::vector<int64_t> out_shape;
        if (a_is_2d && !b_is_2d) {
            out_shape = {mat_a.size(0), mat_b.size(-1)};           // (M, N)
        } else if (!a_is_2d && b_is_2d) {
            out_shape = {mat_a.size(1), mat_b.size(-1)};           // (M, N)
        } else if (a_is_2d && b_is_2d) {
            out_shape = {offs->size(0), mat_a.size(0), mat_b.size(1)};  // (G, M, N)
        } else {
            out_shape = {mat_a.size(0), mat_a.size(1), mat_b.size(-1)}; // (G, M, N)
        }
        at::Tensor out = npu_preparation::apply_tensor_without_format(out_shape, options);
        at::native::_grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
        return out;
#else
        TORCH_CHECK(false,
            "aclnnGroupedMatmulV5 is not available, and native _grouped_mm_fallback "
            "requires PyTorch >= 2.9. Please update CANN to support V5, or upgrade PyTorch.",
            OPS_ERROR(ErrCode::INTERNAL));
#endif
    }

    // 3D x 3D -> aclnnBatchMatMul (via at::bmm, NPU auto-dispatches)
    // Use at::bmm instead of at::bmm_out to avoid parameter order differences
    // across PyTorch versions (old: out first, new: out last)
    if (!a_is_2d && !b_is_2d) {
        at::Tensor out = at::bmm(mat_a, mat_b);
        return out;
    }

    // Rest -> aclnnGroupedMatmulV5
    // Infer split_item and group_type from input dimensions, prepare data, then
    // forward to V5 — same pattern as npu_grouped_matmul (allocate output + call V5).
    auto group_list = prepare_group_list(offs.value());

    int64_t split_item = 0;
    int64_t group_type = 0;
    std::vector<at::Tensor> x_vec;
    std::vector<at::Tensor> weight_vec;
    std::vector<at::Tensor> result_vec;
    bool need_transpose_back = false;

    if (a_is_2d && !b_is_2d) {
        // 2D x 3D: mat_a(M, K) x mat_b(G, K, N) -> out(M, N)
        // M-axis split, direct forwarding — no data transformation needed
        split_item = IN_SPLIT_OUT_SPLIT;
        group_type = M_SPLIT;
        int64_t M = mat_a.size(0);
        int64_t N = mat_b.size(-1);
        x_vec = {mat_a.contiguous()};
        weight_vec = {mat_b.contiguous()};
        auto output_size = op_infer::array_to_small_vector({M, N});
        result_vec = {npu_preparation::apply_tensor_without_format(output_size, options)};
    } else if (!a_is_2d && b_is_2d) {
        // 3D x 2D: mat_a(G, M, K) x mat_b(K, N) -> out(M, N)
        // V5 groupType=0 expects x=2D, weight=3D — swap via transpose
        split_item = IN_SPLIT_OUT_SPLIT;
        group_type = M_SPLIT;
        int64_t N = mat_b.size(-1);
        int64_t M = mat_a.size(1);
        x_vec = {mat_b.transpose(-2, -1).contiguous()};
        weight_vec = {mat_a.transpose(-2, -1).contiguous()};
        auto output_size = op_infer::array_to_small_vector({N, M});
        result_vec = {npu_preparation::apply_tensor_without_format(output_size, options)};
        need_transpose_back = true;
    } else {
        // 2D x 2D: mat_a(M, K) x mat_b(K, N), offs(G,) -> out(G, M, N)
        // K-axis split. V5 groupType=2 requires x non-contiguous (transposed strides)
        // with logical shape (M, K) so dim(1)=K matches weight.dim(0)=K.
        split_item = IN_NOT_SPLIT_OUT_SPLIT;
        group_type = K_SPLIT;
        int64_t M = mat_a.size(0);
        int64_t N = mat_b.size(1);
        int64_t G = offs->size(0);
        x_vec = {mat_a.t().contiguous().t()};
        weight_vec = {mat_b.contiguous()};
        auto output_size = op_infer::array_to_small_vector({G, M, N});
        result_vec = {npu_preparation::apply_tensor_without_format(output_size, options)};
    }

    exec_grouped_matmul_v5(x_vec, weight_vec, group_list, split_item, group_type, result_vec);

    if (need_transpose_back) {
        return result_vec[0].transpose(-2, -1).contiguous();
    }
    return result_vec[0];
}

}  // namespace op_api
