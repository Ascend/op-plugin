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
constexpr int64_t INT4_NUMS_IN_INT32 = 8;
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
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
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
                    short_dim, " and  the long_dim is ", long_dim);
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_val = batch_val * cur_batch_value;
        batch_record.push_back(cur_batch_value);
    }
    return batch_val;
}

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor npu_quant_matmul(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& scale,
                            const c10::optional<at::Tensor>& offset, const c10::optional<at::Tensor>& pertoken_scale,
                            const c10::optional<at::Tensor>& bias, c10::optional<at::ScalarType> output_dtype)
{
    if (is_nz_format(x2)) {
        const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnQuantMatmulWeightNzGetWorkspaceSize");
        const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnQuantMatmulWeightNz");
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                    "Get aclnnQuantMatmulWeightNz or aclnnQuantMatmulWeightNzGetWorkspaceSize faild, only "
                    "aclnnQuantMatmulWeightNz support X2's format is nz, please upgrade CANN.",
                    OPS_ERROR(ErrCode::PARAM));
    }

    bool is_a4w4 = x1.dtype() == at::kInt && x2.dtype() == at::kInt;
    bool trans_x2 = is_transpose_last_two_dims(x2);
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto x1_k_dim = x1.size(x1_dim_num - 1);
    auto x2_n_dim = (is_a4w4 && !trans_x2) ? x2.size(x2_dim_num - 1) * INT4_NUMS_IN_INT32 : x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - LAST_SECOND_DIM_INDEX);

    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape(x1, x2, batch_record);
    const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x1 : x2;
    auto output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }
    c10::TensorOptions options;
    if (!output_dtype.has_value() ||  output_dtype.value() == at::kChar) {
        options = x1.options().dtype(at::kChar);
    } else if (output_dtype == at::kHalf) {
        options = x1.options().dtype(at::kHalf);
    } else if (output_dtype == at::kBFloat16) {
        options = x1.options().dtype(at::kBFloat16);
    } else if (output_dtype == at::kInt) {
        options = x1.options().dtype(at::kInt);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    bool transpose1 = false;
    bool transpose2 = false;

    if (scale.dtype() == at::kFloat && !pertoken_scale.has_value() && output_dtype != at::kBFloat16 &&
        output_dtype != at::kInt) {
        const at::Tensor quant_param = op_api::npu_trans_quant_param(scale, offset);
        if (!is_a4w4 && is_nz_format(x2)) {
            at::Tensor yscale = at::empty({0}, options);
            at::Tensor x1Offset = at::empty({0}, options);
            at::Tensor yOffset = at::empty({0}, options);
            int64_t groupSize = 0;
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1, x2, pertoken_scale_real, quant_param, yscale, x1Offset,
                         offset_real, yOffset, bias_real, transpose1, transpose2, groupSize, result);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV4, x1, x2, quant_param, offset_real, pertoken_scale_real, bias_real,
                         transpose1, transpose2, result);
        }
    } else {
        if (!is_a4w4 && is_nz_format(x2)) {
            at::Tensor yscale = at::empty({0}, options);
            at::Tensor x1Offset = at::empty({0}, options);
            at::Tensor yOffset = at::empty({0}, options);
            int64_t groupSize = 0;
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1, x2, pertoken_scale_real, scale, yscale, x1Offset, offset_real,
                         yOffset, bias_real, transpose1, transpose2, groupSize, result);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV4, x1, x2, scale, offset_real, pertoken_scale_real, bias_real,
                         transpose1, transpose2, result);
        }
    }
    return result;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor npu_quant_matmul(const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& scale,
                            const c10::optional<at::Tensor>& offset, const c10::optional<at::Tensor>& bias,
                            c10::optional<c10::string_view> output_dtype)
{
    bool is_a4w4 = x1.dtype() == at::kInt && x2.dtype() == at::kInt;
    bool trans_x2 = is_transpose_last_two_dims(x2);
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto x1_k_dim = x1.size(x1_dim_num - 1);
    auto x2_n_dim = (is_a4w4 && !trans_x2) ? x2.size(x2_dim_num - 1) * INT4_NUMS_IN_INT32 : x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - 2);

    std::vector<uint64_t> batch_record;
    uint64_t batch_val = infer_out_batch_shape(x1, x2, batch_record);
    const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x1 : x2;
    auto output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (size_t i = 0; i < long_tensor.dim() - LAST_SECOND_DIM_INDEX; i++) {
        output_size[i] = batch_record[i];
    }
    c10::TensorOptions options;
    if (!output_dtype.has_value() ||  *output_dtype == "int8") {
        options = x1.options().dtype(at::kChar);
    } else if (*output_dtype == "float16") {
        options = x1.options().dtype(at::kHalf);
    } else if (*output_dtype == "bfloat16") {
        options = x1.options().dtype(at::kBFloat16);
    }
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    const at::Tensor &offset_real = offset.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    bool transpose1 = false;
    bool transpose2 = false;

    if (scale.dtype() == at::kFloat) {
        const at::Tensor quant_param = op_api::npu_trans_quant_param(scale, offset);
        if (!is_a4w4 && is_nz_format(x2)) {
            at::Tensor x1scale = at::empty({0}, options);
            at::Tensor yscale = at::empty({0}, options);
            at::Tensor x1Offset = at::empty({0}, options);
            at::Tensor yOffset = at::empty({0}, options);
            int64_t groupSize = 0;
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1, x2, x1scale, quant_param, yscale, x1Offset, offset_real, yOffset,
                         bias_real, transpose1, transpose2, 0, result);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV3, x1, x2, quant_param, offset_real, bias_real, transpose1, transpose2, result);
        }
    } else {
        if (!is_a4w4 && is_nz_format(x2)) {
            at::Tensor x1scale = at::empty({0}, options);
            at::Tensor yscale = at::empty({0}, options);
            at::Tensor x1Offset = at::empty({0}, options);
            at::Tensor yOffset = at::empty({0}, options);
            int64_t groupSize = 0;
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1, x2, x1scale, scale, yscale, x1Offset, offset_real, yOffset,
                         bias_real, transpose1, transpose2, 0, result);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV3, x1, x2, scale, offset_real, bias_real, transpose1, transpose2, result);
        }
    }
    return result;
}
#endif

}