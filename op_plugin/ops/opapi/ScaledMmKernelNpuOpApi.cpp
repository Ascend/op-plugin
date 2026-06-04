// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

static bool is_transpose_certain_two_dims(const at::Tensor &tensor, int64_t dim)
{
    return tensor.stride(dim + 1) == tensor.stride(dim) * tensor.size(dim);
}

static bool is_x_scale_same_transpose(const at::Tensor &x, const at::Tensor &scale, int64_t dim_x, int64_t dim_scale)
{
    if (x.dim() < dim_x + 2 || scale.dim() < dim_scale + 2) { // make sure dims after the start dim are no less than 2
        return true;
    }
    if (x.size(dim_x) == 1 && x.size(dim_x + 1)== 1) {
        return true;
    }
    if (scale.size(dim_scale) == 1 && scale.size(dim_scale + 1)== 1) {
        return true;
    }
    bool x_trans = is_transpose_certain_two_dims(x, dim_x);
    bool scale_trans = is_transpose_certain_two_dims(scale, dim_scale);
    if (x_trans == scale_trans) {
        return true;
    }
    return false;
}
static bool is_nz_format(const at::Tensor& x2)
{
    const torch_npu::NPUStorageDesc &tensor_desc =
        torch_npu::NPUBridge::GetNpuStorageImpl(x2)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ ||
        tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_4 ||
        tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ_C0_16;
}

static uint64_t infer_out_batch_shape(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
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

static int64_t check_and_get_groups(at::IntArrayRef group_size_list)
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

static const std::map<c10::ScalarType, int64_t> SCALAR_TO_INT_TYPE_MAP = {
    {c10::ScalarType::Char, static_cast<int64_t>(at::kChar)},                    /**< int8 */
    {c10::ScalarType::Int, static_cast<int64_t>(at::kInt)},                      /**< int32 */
    {c10::ScalarType::BFloat16, static_cast<int64_t>(at::kBFloat16)},            /**< bfp16 */
    {c10::ScalarType::Half, static_cast<int64_t>(at::kHalf)},                    /**< fp16 */
    {c10::ScalarType::Float, static_cast<int64_t>(at::kFloat)},                  /**< fp32 */

    {c10::ScalarType::Float8_e4m3fn, static_cast<int64_t>(at::kFloat8_e4m3fn)},  /**< fp8e4m3 */
    {c10::ScalarType::Float8_e5m2, static_cast<int64_t>(at::kFloat8_e5m2)},      /**< fp8e5m2 */
    {c10::ScalarType::Byte, static_cast<int64_t>(c10_npu::DType::HIFLOAT8)}      /**< hif8 */
};

static c10::optional<int64_t> ToIntType(const std::optional<c10::ScalarType> &torchType) {
    c10::optional<int64_t> int_type = c10::nullopt;
    if (torchType.has_value()) {
        const auto &it = SCALAR_TO_INT_TYPE_MAP.find(torchType.value());
        if (it != SCALAR_TO_INT_TYPE_MAP.cend()) {
            int_type = c10::make_optional(it->second);
        }
    }
    return int_type;
}

at::Tensor _scaled_mm(const at::Tensor &mat_a,
    const at::Tensor &mat_b,
    const at::Tensor &scale_a,
    const at::Tensor &scale_b,
    const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& scale_result, // 对result做scale，仅当output为float8时才有用，
    std::optional<c10::ScalarType> out_dtype,   // 当前不支持float8
    bool use_fast_accum)

{

    // check A5
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
                "_scaled_mm is supported only on the Ascend950 platform and after.", OPS_ERROR(ErrCode::PARAM));

    // Check data types: mat_a and mat_b must be float8 type
    TORCH_CHECK(mat_a.scalar_type() == c10::ScalarType::Float8_e4m3fn ||
                 mat_a.scalar_type() == c10::ScalarType::Float8_e5m2,
        "mat_a must be float8 type (Float8_e4m3fn or Float8_e5m2), but got ", mat_a.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mat_b.scalar_type() == c10::ScalarType::Float8_e4m3fn ||
                 mat_b.scalar_type() == c10::ScalarType::Float8_e5m2,
        "mat_b must be float8 type (Float8_e4m3fn or Float8_e5m2), but got ", mat_b.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));

    // Check multiplication of two Float8_e5m2 is not supported (reference: _scaled_mm_out_cuda)
    TORCH_CHECK(mat_a.scalar_type() != c10::ScalarType::Float8_e5m2 ||
                 mat_b.scalar_type() != c10::ScalarType::Float8_e5m2,
        "Multiplication of two Float8_e5m2 matrices is not supported",
        OPS_ERROR(ErrCode::TYPE));

    // Check data types: scale_a and scale_b must be float32 or float8_e8m0 type
    TORCH_CHECK(scale_a.scalar_type() == c10::ScalarType::Float ||
                 scale_a.scalar_type() == npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(c10_npu::DType::FLOAT8_E8M0)),
        "scale_a must be float32 or float8_e8m0 type, but got ", scale_a.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(scale_b.scalar_type() == c10::ScalarType::Float ||
                 scale_b.scalar_type() == npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(c10_npu::DType::FLOAT8_E8M0)),
        "scale_b must be float32 or float8_e8m0 type, but got ", scale_b.scalar_type(),
        OPS_ERROR(ErrCode::TYPE));

    // Check bias (reference: _scaled_mm_out_cuda)
    if (bias.has_value()) {
        TORCH_CHECK(bias->numel() == mat_b.sizes()[1],
            "Bias must be size ", mat_b.sizes()[1], " but got ", bias->numel(),
            OPS_ERROR(ErrCode::PARAM));
        // Check out_dtype vs bias compatibility
        auto out_dtype_value = out_dtype.value_or(c10::ScalarType::BFloat16);
        TORCH_CHECK(out_dtype_value != c10::ScalarType::Float,
            "Bias is not supported when out_dtype is set to Float32",
            OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(bias->scalar_type() == c10::ScalarType::BFloat16 ||
                    bias->scalar_type() == c10::ScalarType::Half,
            "Bias must be BFloat16 or Half, but got ", bias->scalar_type(),
            OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK((out_dtype_value != c10::ScalarType::Float &&
                     out_dtype_value != c10::ScalarType::BFloat16) ||
                    bias->scalar_type() == c10::ScalarType::BFloat16,
            "Bias must be BFloat16 to compute ", out_dtype_value,
            " output, but got ", bias->scalar_type(),
            OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(out_dtype_value != c10::ScalarType::Half ||
                    bias->scalar_type() == c10::ScalarType::Half,
            "Bias must be Float16 to compute ", out_dtype_value,
            " output, but got ", bias->scalar_type(),
            OPS_ERROR(ErrCode::TYPE));
    }


    // Check scale_result: currently only supports null/empty because output float8 is not supported
    TORCH_CHECK(!scale_result.has_value() || scale_result == c10::nullopt,
        "scale_result is not supported currently, as output float8 type is not enabled. "
        "scale_result is only utilized when output is float8 type.",
        OPS_ERROR(ErrCode::NOT_SUPPORT));

    // Check out_dtype: currently only supports Float32, BFloat16, and Float16
    if (out_dtype.has_value()) {
        TORCH_CHECK(out_dtype.value() == c10::ScalarType::Float ||
                     out_dtype.value() == c10::ScalarType::BFloat16 ||
                     out_dtype.value() == c10::ScalarType::Half,
            "out_dtype must be Float32, BFloat16, or Float16, but got ", out_dtype.value(),
            OPS_ERROR(ErrCode::TYPE));
    }


    // Check sizes
    TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a matrix, please check mat_a dim num." ,OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a matrix, please check mat_b dim num." ,OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[0], "mat_a and mat_b shapes cannot be multiplied (",mat_a.sizes()[0],
        "x", mat_b.sizes()[1], " and ", mat_b.sizes()[0], "x", mat_b.sizes()[1], ")", OPS_ERROR(ErrCode::PARAM));


    //////////////rowwise transfer//////////////
    at::Tensor processed_scale_a = scale_a;
    at::Tensor processed_scale_b = scale_b;
    if (mat_a.scalar_type() == c10::ScalarType::Float8_e4m3fn
        && mat_b.scalar_type() == c10::ScalarType::Float8_e4m3fn
        && scale_a.scalar_type() == c10::ScalarType::Float
        && scale_b.scalar_type() == c10::ScalarType::Float) {
        if (scale_a.dim() == 2 && scale_a.sizes()[1] == 1 && scale_b.dim() == 2 && scale_b.sizes()[0] == 1) {
            TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be contiguous in last dim");
            TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be contiguous in first dim");
            processed_scale_a = scale_a.squeeze(-1).contiguous();
            processed_scale_b = scale_b.squeeze(0);
        }
    }

    //////////////parameters transfer//////////////
    at::Tensor x1 = mat_a;
    at::Tensor x2 = mat_b;
    c10::optional<at::Tensor> pertoken_scale = processed_scale_a;
    at::Tensor scale = processed_scale_b;
    c10::optional<int64_t> scale_dtype = c10::nullopt;   // scale_b.scalar_type()

    c10::optional<int64_t> output_dtype = ToIntType(out_dtype);
    c10::optional<at::Tensor> offset = c10::nullopt;
    c10::optional<int64_t> x1_dtype = c10::nullopt;
    c10::optional<int64_t> x2_dtype = c10::nullopt;
    c10::optional<int64_t> pertoken_scale_dtype = c10::nullopt;
    c10::OptionalIntArrayRef group_sizes = c10::nullopt;
    c10::optional<at::Tensor> y_scale = c10::nullopt;


///////////////////////////////////////////npu_quant_matmul//////////////////////////////////////
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
    int64_t group_size = check_and_get_groups(group_size_list);
    bool is_a4w4 = x1.dtype() == at::kInt && x2.dtype() == at::kInt;
    bool trans_x1 = is_transpose_last_two_dims(x1);
    bool trans_x2 = is_transpose_last_two_dims(x2);
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto x2_n_dim = (is_a4w4 && !trans_x2) ? x2.size(x2_dim_num - 1) * INT4_NUMS_IN_INT32 : x2.size(x2_dim_num - 1);

#if VERSION_BETWEEN(V2R1, V2R7)
    bool mxfp4_valid = x1_dtype.has_value() && x2_dtype.has_value() &&
        x1_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) &&
        x2_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1);
#endif
#if VERSION_BETWEEN(V2R8, VERSION_NEWEST)
    bool mxfp4_valid = false;
    if (x1_dtype.has_value()) {
        mxfp4_valid = x1_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1);
    } else {
        mxfp4_valid = x1.scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    }
    if (x2_dtype.has_value()) {
        mxfp4_valid = mxfp4_valid && x2_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1);
    } else {
        mxfp4_valid = mxfp4_valid && x2.scalar_type() == at::ScalarType::Float4_e2m1fn_x2;
    }
#endif

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
        if (mxfp4_valid) {
            TORCH_CHECK(x1.dim() >= 2 && x1.dim() <= 6,
                "x1 dim num should be 2 ~ 6, please check x1 dim num. Actual x1 dim = ", x1.dim(),
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(x2.dim() >= 2 && x2.dim() <= 6,
                "x2 dim num should be 2 ~ 6, please check x2 dim num. Actual x2 dim = ", x2.dim(),
                OPS_ERROR(ErrCode::PARAM));
            int64_t x1_size_last_second = x1.sizes()[x1_dim_num - LAST_SECOND_DIM_INDEX];
            int64_t x2_size_last = x2.sizes()[x2_dim_num - 1];
            int64_t real_m = !trans_x1 ? x1_size_last_second : x1_size_last_second * FP4_IN_INT8;
            int64_t real_n = trans_x2 ? x2_size_last : x2_size_last * FP4_IN_INT8;
            output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = real_m;
            output_size[long_tensor.dim() - 1] = real_n;
        } else {
            output_size[long_tensor.dim() - LAST_SECOND_DIM_INDEX] = x1.size(x1_dim_num - LAST_SECOND_DIM_INDEX);
            output_size[long_tensor.dim() - 1] = x2_n_dim;
        }
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
    at::Tensor x2_offset = at::Tensor();
    at::Tensor y_offset = at::empty({0}, options);
    if (is_a8W4_int) {  // Only A8W4 int needs y_offset
        y_offset = offset_real;
    } else {
        x2_offset = offset_real;
    }

    bool use_aclnn_v5 = x1_dtype.has_value() || (x1.dtype() != at::kInt && x1.dtype() != at::kChar) ||
        is_a8W4_float || is_a8W4_int;

    aclDataType pertoken_scale_dtype_real = pertoken_scale_dtype.has_value()
        ? c10_npu::GetAclDataType(pertoken_scale_dtype.value())
        : (pertoken_scale.has_value()
                  ? c10_npu::GetAclDataType(static_cast<int64_t>(pertoken_scale_real.scalar_type()))
                  : aclDataType::ACL_INT8);
    bool need_check_trans = pertoken_scale.has_value()
        && (((pertoken_scale_real.dim() == x1.dim() && scale.dim() == x2.dim())
                || pertoken_scale_dtype_real == aclDataType::ACL_FLOAT8_E8M0)
               && (pertoken_scale_real.dim() >= 2 && scale.dim() >= 2))
        && !(is_a8W4_float || is_a8W4_int);
    if (need_check_trans) {
        int64_t dim_x1 = x1.dim() - 2; //  check the last 2 dim
        int64_t dim_x2 = x2.dim() - 2; //  check the last 2 dim
        int64_t dim_x1_scale = 0;
        int64_t dim_x2_scale = 0;
        if (pertoken_scale_dtype_real != aclDataType::ACL_FLOAT8_E8M0) {
            dim_x1_scale = pertoken_scale_real.dim() - 2; // check the last 2 dim in GB/BB
            dim_x2_scale = scale.dim() - 2; // check the last 2 dim in GB/BB
        }
        TORCH_CHECK(is_x_scale_same_transpose(x1, pertoken_scale_real, dim_x1, dim_x1_scale),
            "Input x1 tensor and pertoken_scale tensor's transpose are not same, please check input.",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(is_x_scale_same_transpose(x2, scale, dim_x2, dim_x2_scale),
            "Input x2 tensor and scale tensor's transpose are not same, please check input.",
            OPS_ERROR(ErrCode::PARAM));
    }

    bool use_trans_quant_param = scale.dtype() == at::kFloat && !pertoken_scale.has_value() &&
        (output_acltype != ACL_BF16 || use_aclnn_v5) && output_acltype != ACL_INT32;
    if (use_trans_quant_param) {
        const at::Tensor quant_param = op_api::npu_trans_quant_param(scale, offset);
        if (is_nz_format(x2)) {
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1_wrapper, x2_wrapper, pertoken_scale_real, quant_param, y_scale,
                x1_offset, x2_offset, y_offset, bias_real, transpose1, transpose2, group_size, result_wrapper);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, pertoken_scale_real, quant_param, y_scale,
                x1_offset, x2_offset, y_offset, bias_real, transpose1, transpose2, group_size, result_wrapper);
        }
    } else {
        if (!is_a4w4 && is_nz_format(x2)) {
            EXEC_NPU_CMD(aclnnQuantMatmulWeightNz, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, y_scale,
                x1_offset, x2_offset, y_offset, bias_real, transpose1, transpose2, group_size, result_wrapper);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulV5, x1_wrapper, x2_wrapper, x1_scale_wrapper, x2_scale_wrapper, y_scale,
                x1_offset, x2_offset, y_offset, bias_real, transpose1, transpose2, group_size, result_wrapper);
        }
    }

    return result;
}

}