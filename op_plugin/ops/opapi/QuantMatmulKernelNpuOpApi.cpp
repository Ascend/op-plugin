// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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
#include <vector>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace op_api {
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr int64_t PERGROUP_DIM_NUM = 2;
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
static const uint64_t GROUP_MAX = 65535UL;
static const size_t A8W4_GROUP_DIM = 3;
static const size_t A8W4_INPUT_DIM = 2;
using npu_preparation = at_npu::native::OpPreparation;

bool static is_transpose_last_two_dims(const at::Tensor &tensor)
{
    if (tensor.dim() < 2 || tensor.dim() > 6) {
        return false;
    }
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        int64_t tmpNxD = tensor.size(dim1) * tensor.size(dim2);
        for (int64_t batchDim = tensor.dim() - 3; batchDim >= 0; batchDim--) {
            if (tensor.stride(batchDim) != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor.size(batchDim);
        }
        if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

static bool is_nz_format(const at::Tensor& x2)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
           tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4;
}

uint64_t infer_out_batch_shape(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
{
    TORCH_CHECK(at_npu::native::FormatHelper::IsBaseFormatType(x2) || is_nz_format(x2),
                "x2 should be in the original image format or nz format, but it is ",
                npu_preparation::get_tensor_npu_format(x2), OPS_ERROR(ErrCode::PARAM));
    uint64_t batch_val = 1;
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto out_dim_num = std::max(x1_dim_num, x2_dim_num);
    auto &shape_long = x1_dim_num > x2_dim_num ? x1 : x2;
    auto &shape_short = x1_dim_num > x2_dim_num ? x2 : x1;
    int64_t vaild_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (int64_t i = 0; i < out_dim_num - LAST_SECOND_DIM_INDEX; i++) {
        auto short_dim = i < vaild_offset ? 1 : shape_short.size(i - vaild_offset);
        auto long_dim = shape_long.size(i);
        TORCH_CHECK(!(short_dim > 1 && long_dim > 1 && short_dim != long_dim),
                    "the x1 shape and x2 shape not supported for broadcast, the short_dim is ",
                    short_dim, " and  the long_dim is ", long_dim, OPS_ERROR(ErrCode::PARAM));
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_val = batch_val * cur_batch_value;
        batch_record.push_back(cur_batch_value);
    }
    return batch_val;
}

int64_t check_and_get_groups(
    at::IntArrayRef group_size_list,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& pertoken_scale)
{
    int64_t groups = 0;
    if (group_size_list.empty()) {
        return groups;
    }
    size_t group_dim = group_size_list.size();
    TORCH_CHECK(group_dim == A8W4_GROUP_DIM, "group_sizes only support input with three elements, but got ",
                group_dim, OPS_ERROR(ErrCode::PARAM));
    int64_t group_m = static_cast<int64_t>(group_size_list[0]);
    int64_t group_n = static_cast<int64_t>(group_size_list[1]);
    int64_t group_k = static_cast<int64_t>(group_size_list[2]);
    bool invalid_group_param = ((group_m <= GROUP_MAX && group_m >= 0)
                                && (group_n <= GROUP_MAX && group_n >= 0)
                                && (group_k <= GROUP_MAX && group_k >= 0));
    TORCH_CHECK(invalid_group_param, "group param value must conform to range [0, 65535]", OPS_ERROR(ErrCode::VALUE));
    groups = static_cast<int64_t>((static_cast<uint64_t>(group_m) << 32) + (static_cast<uint64_t>(group_n) << 16) +
                                  (static_cast<uint64_t>(group_k)));
    return groups;
}

at::Tensor npu_quant_matmul(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &scale,
                            const c10::optional<at::Tensor> &offset, const c10::optional<at::Tensor> &pertoken_scale,
                            const c10::optional<at::Tensor> &bias, c10::optional<int64_t> output_dtype,
                            c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype,
                            c10::optional<int64_t> pertoken_scale_dtype, c10::optional<int64_t> scale_dtype,
                            c10::OptionalIntArrayRef group_sizes, const c10::optional<at::Tensor> &y_scale)
{
    if (is_nz_format(x2)) {
        static const bool is_quant_matmul_weight_nz_available = check_aclnn_kernel_available("aclnnQuantMatmulWeightNz");
        TORCH_CHECK(is_quant_matmul_weight_nz_available,
                    "Get aclnnQuantMatmulWeightNz or aclnnQuantMatmulWeightNzGetWorkspaceSize failed, only "
                    "aclnnQuantMatmulWeightNz support X2's format is nz, please upgrade CANN.",
                    OPS_ERROR(ErrCode::PARAM));
    } else {
        static const bool is_quant_matmul_v5_available = check_aclnn_kernel_available("aclnnQuantMatmulV5");
        TORCH_CHECK(is_quant_matmul_v5_available,
                    "Get aclnnQuantMatmulV5 or aclnnQuantMatmulV5 failed, only "
                    "aclnnQuantMatmulV5 support A8W4, please upgrade CANN.",
                    OPS_ERROR(ErrCode::TYPE));
    }
    bool is_a8W4_int = x1.dtype() == at::kChar && x2.dtype() == at::kInt;
    bool is_a8W4_float = x1.dtype() == at::kFloat8_e4m3fn && x2.dtype() == at::kFloat;
    at::IntArrayRef group_size_list = group_sizes.value_or(at::IntArrayRef{});
    int64_t group_size = check_and_get_groups(group_size_list, x1, x2, scale, pertoken_scale);
    bool is_a4w4 = x1.dtype() == at::kInt && x2.dtype() == at::kInt;
    bool trans_x2 = is_transpose_last_two_dims(x2);
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto x2_n_dim = (is_a4w4 && !trans_x2) ? x2.size(x2_dim_num - 1) * INT4_NUMS_IN_INT32 : x2.size(x2_dim_num - 1);

    c10::SmallVector<int64_t, SIZE> output_size;
    if (is_a8W4_int) {
        output_size = {x1.sizes()[0], x2.sizes()[1] * INT4_NUMS_IN_INT32};
    } else if (is_a8W4_float) {
        if (trans_x2) {
            output_size = {x1.sizes()[0], x2.sizes()[1]};
        } else {
            output_size = {x1.sizes()[0], x2.sizes()[1] * INT4_NUMS_IN_INT32};
        }
    } else {
        std::vector<uint64_t> batch_record;
        uint64_t batch_val = infer_out_batch_shape(x1, x2, batch_record);
        const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x1 : x2;
        output_size = op_infer::array_to_small_vector(long_tensor.sizes());
        output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
        output_size[long_tensor.dim() - 1] = x2_n_dim;
        for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
            output_size[i] = static_cast<int64_t>(batch_record[i]);
        }
    }
    c10::TensorOptions options;
    aclDataType output_acltype = ACL_INT8;
    if (!output_dtype.has_value()) {
        options = x1.options().dtype(at::kChar);
    } else {
        output_acltype = c10_npu::GetAclDataType(output_dtype.value());
        options = x1.options().dtype(npu_preparation::convert_to_scalar_type(output_acltype));
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    bool transpose1 = false;
    bool transpose2 = false;

    TensorWrapper x1_wrapper = make_wrapper(x1, x1_dtype);
    TensorWrapper x2_wrapper = make_wrapper(x2, x2_dtype);
    TensorWrapper x1_scale_wrapper = make_wrapper(pertoken_scale_real, pertoken_scale_dtype);
    TensorWrapper x2_scale_wrapper = make_wrapper(scale, scale_dtype);
    TensorWrapper result_wrapper = make_wrapper(result, output_dtype);
    at::Tensor x1_offset = at::empty({0}, options);
    at::Tensor y_offset = at::empty({0}, options);

    bool use_aclnn_v5 = x1_dtype.has_value() || (x1.dtype() != at::kInt && x1.dtype() != at::kChar) ||
         is_a8W4_float || is_a8W4_int;
    bool use_trans_quant_param = scale.dtype() == at::kFloat && !pertoken_scale.has_value() &&
                                 (output_acltype != ACL_BF16 || use_aclnn_v5) && output_acltype != ACL_INT32;
    if (use_trans_quant_param) {
        const at::Tensor quant_param = op_api::npu_trans_quant_param(scale, offset);
        if (is_nz_format(x2)) {
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1_wrapper, x2_wrapper, pertoken_scale_real, quant_param, y_scale,
                         x1_offset, offset_real, y_offset, bias_real, transpose1, transpose2, group_size,
                         result_wrapper);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, pertoken_scale_real, quant_param, y_scale,
                         x1_offset, offset_real, y_offset, bias_real, transpose1, transpose2, group_size,
                         result_wrapper);
        }
    } else {
        if (!is_a4w4 && is_nz_format(x2)) {
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, y_scale,
                         x1_offset, offset_real, y_offset, bias_real, transpose1, transpose2, group_size,
                         result_wrapper);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, y_scale,
                         x1_offset, offset_real, y_offset, bias_real, transpose1, transpose2, group_size,
                         result_wrapper);
        }
    }

    return result;
}
}