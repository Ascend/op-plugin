// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include <unordered_map>

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t NZ_16 = 16;
constexpr int64_t NZ_C0_16 = 16;
constexpr int64_t DIMS_2 = 2;
constexpr int64_t DIMS_3 = 3;
constexpr int64_t DIMS_4 = 4;
constexpr int64_t DIMS_MAX = 5;
constexpr int64_t IDX_0 = 0;
constexpr int64_t IDX_1 = 1;
constexpr int64_t IDX_2 = 2;
constexpr int64_t IDX_3 = 3;
constexpr int64_t MX_SCALE_LAST_DIM = 2;

struct QuantContext {
    const at::Tensor &weight;
    const at::Tensor &weight_scale;
    const c10::optional<at::Tensor> &weight_offset;
    const c10::optional<at::Tensor> &bias;
    int64_t x_dtype;
    int64_t weight_dtype;
    c10::optional<int64_t> x_scale_dtype;
    int64_t weight_scale_dtype;
    int64_t k_group_size;
    c10_npu::SocVersion soc_version;
    bool is_weight_trans = false;

    at::Tensor out_weight;
    at::Tensor out_weight_scale;
    at::Tensor out_weight_offset;
    at::Tensor out_bias;
};

using PrepareFunc = void (*)(QuantContext &ctx);
using DataFlowJudgeFunc = bool (*)(QuantContext &ctx);
using DataFlowConfig = std::pair<DataFlowJudgeFunc, std::vector<PrepareFunc>>;

inline int64_t ceil_div(int64_t a, int64_t b) {
    TORCH_CHECK(b != 0, "Division by zero in ceil_div." + OPS_ERROR(ErrCode::VALUE));
    return (a + b - 1) / b;
}

static bool is_transpose_certain_two_dims(const at::Tensor &tensor, int64_t first_dim) {
    TORCH_CHECK(first_dim >= 0 && first_dim + 1 < tensor.dim(), "first_dim out of bounds: first_dim=", first_dim,
        ", tensor.dim()=", tensor.dim(), OPS_ERROR(ErrCode::PARAM));
    return tensor.stride(first_dim + 1) == tensor.stride(first_dim) * tensor.size(first_dim);
}

bool judge_mm_mx_a8w4(QuantContext &ctx) {
    aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
    aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
    aclDataType x_scale_acl_dtype =
        ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
    aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);

    OP_LOG_DEBUG("judge_mm_mx_a8w4: x_acl_dtype=%d, weight_acl_dtype=%d, x_scale_acl_dtype=%d, "
                 "weight_scale_acl_dtype=%d, weight_dim=%d",
        static_cast<int>(x_acl_dtype), static_cast<int>(weight_acl_dtype), static_cast<int>(x_scale_acl_dtype),
        static_cast<int>(weight_scale_acl_dtype), static_cast<int>(ctx.weight.dim()));

    bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) && (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
        (x_scale_acl_dtype == ACL_FLOAT8_E8M0) && (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
    if (dtype_match && ctx.weight.dim() == DIMS_2) {
        TORCH_CHECK(ctx.weight_scale.dim() == DIMS_3, "Input weight scale tensor should be 3D, but got ",
            ctx.weight_scale.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(is_transpose_certain_two_dims(ctx.weight, 0),
            "Input weight tensor should be transposed, please check input.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(is_transpose_certain_two_dims(ctx.weight_scale, 0),
            "Input weight scale tensor should be transposed, please check input.", OPS_ERROR(ErrCode::PARAM));
        ctx.is_weight_trans = true;
        return true;
    }
    return false;
}

bool judge_gmm_mx_a8w4(QuantContext &ctx) {
    aclDataType x_acl_dtype = c10_npu::GetAclDataType(ctx.x_dtype);
    aclDataType weight_acl_dtype = c10_npu::GetAclDataType(ctx.weight_dtype);
    aclDataType x_scale_acl_dtype =
        ctx.x_scale_dtype.has_value() ? c10_npu::GetAclDataType(ctx.x_scale_dtype.value()) : ACL_DT_UNDEFINED;
    aclDataType weight_scale_acl_dtype = c10_npu::GetAclDataType(ctx.weight_scale_dtype);

    OP_LOG_DEBUG("judge_gmm_mx_a8w4: x_acl_dtype=%d, weight_acl_dtype=%d, x_scale_acl_dtype=%d, "
                 "weight_scale_acl_dtype=%d, weight_dim=%d",
        static_cast<int>(x_acl_dtype), static_cast<int>(weight_acl_dtype), static_cast<int>(x_scale_acl_dtype),
        static_cast<int>(weight_scale_acl_dtype), static_cast<int>(ctx.weight.dim()));

    bool dtype_match = (x_acl_dtype == ACL_FLOAT8_E4M3FN) && (weight_acl_dtype == ACL_FLOAT4_E2M1) &&
        (x_scale_acl_dtype == ACL_FLOAT8_E8M0) && (weight_scale_acl_dtype == ACL_FLOAT8_E8M0);
    if (dtype_match && ctx.weight.dim() == DIMS_3) {
        TORCH_CHECK(ctx.weight_scale.dim() == DIMS_4, "Input weight scale tensor should be 4D, but got ",
            ctx.weight_scale.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(is_transpose_certain_two_dims(ctx.weight, 1),
            "Input weight tensor should be transposed, please check input.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(is_transpose_certain_two_dims(ctx.weight_scale, 1),
            "Input weight scale tensor should be transposed, please check input.", OPS_ERROR(ErrCode::PARAM));
        ctx.is_weight_trans = true;
        return true;
    }
    return false;
}

template <bool IsGmm, int64_t NzC0, aclFormat OutWeightFormat> static void prepare_out_weight_nz(QuantContext &ctx) {
    auto weight_sizes = ctx.weight.sizes();
    int64_t weight_dim = ctx.weight.dim();
    int64_t k = weight_sizes[weight_dim - IDX_2];
    int64_t n = weight_sizes[weight_dim - IDX_1];

    c10::SmallVector<int64_t, DIMS_MAX> storage_shape;
    c10::SmallVector<int64_t, DIMS_MAX> trans_stride;

    if (ctx.is_weight_trans) {
        if constexpr (IsGmm) {
            int64_t group_num = weight_sizes[IDX_0];
            storage_shape = {group_num, ceil_div(k, NzC0), ceil_div(n, NZ_16), NZ_16, NzC0};
            trans_stride = {k * n, 1, k};
        } else {
            storage_shape = {ceil_div(k, NzC0), ceil_div(n, NZ_16), NZ_16, NzC0};
            trans_stride = {1, k};
        }
    } else {
        if constexpr (IsGmm) {
            int64_t group_num = weight_sizes[IDX_0];
            storage_shape = {group_num, ceil_div(n, NzC0), ceil_div(k, NZ_16), NZ_16, NzC0};
        } else {
            storage_shape = {ceil_div(n, NzC0), ceil_div(k, NZ_16), NZ_16, NzC0};
        }
    }

    int64_t storage_size = 1;
    for (auto dim : storage_shape) {
        storage_size *= dim;
    }

    ctx.out_weight = npu_preparation::apply_tensor_without_format({storage_size}, ctx.weight.options());

    auto weight_view_shape = op_infer::array_to_small_vector(ctx.weight.sizes());

    if (ctx.is_weight_trans) {
        ctx.out_weight = ctx.out_weight.as_strided_(weight_view_shape, trans_stride);
    } else {
        ctx.out_weight.unsafeGetTensorImpl()->set_sizes_contiguous(weight_view_shape);
    }

    at_npu::native::StorageDescHelper::SetDesc(ctx.out_weight, ctx.out_weight.sizes(), storage_shape,
        ctx.out_weight.strides(), static_cast<aclFormat>(OutWeightFormat));
}

template <bool IsGmm> static void prepare_out_weight_scale_mx(QuantContext &ctx) {
    constexpr int64_t expected_dim = IsGmm ? DIMS_4 : DIMS_3;
    TORCH_CHECK(ctx.weight_scale.dim() == expected_dim, "Input weight scale tensor should be ", expected_dim,
        "D, but got ", ctx.weight_scale.dim(), OPS_ERROR(ErrCode::PARAM));

    auto scale_view_shape = op_infer::array_to_small_vector(ctx.weight_scale.sizes());
    ctx.out_weight_scale = npu_preparation::apply_tensor_without_format(scale_view_shape, ctx.weight_scale.options());

    if (ctx.is_weight_trans) {
        int64_t scale_dim = ctx.weight_scale.dim();
        int64_t k_scale = scale_view_shape[scale_dim - IDX_3];
        int64_t n = scale_view_shape[scale_dim - IDX_2];
        c10::SmallVector<int64_t, DIMS_MAX> trans_stride; // 对应内存转置存储
        if constexpr (IsGmm) {
            trans_stride = {MX_SCALE_LAST_DIM * k_scale * n, MX_SCALE_LAST_DIM, MX_SCALE_LAST_DIM * k_scale,
                1}; // 对应内存中的 [group_num, n, k_scale, 2]
        } else {
            trans_stride = {MX_SCALE_LAST_DIM, MX_SCALE_LAST_DIM * k_scale, 1}; // 对应内存中的 [n, k_scale, 2]
        }
        ctx.out_weight_scale = ctx.out_weight_scale.as_strided_(scale_view_shape, trans_stride);
    }
}

static void prepare_out_weight_offset(QuantContext &ctx) {
    if (ctx.weight_offset.has_value() && ctx.weight_offset.value().defined()) {
        auto offset_sizes = op_infer::array_to_small_vector(ctx.weight_offset.value().sizes());
        ctx.out_weight_offset =
            npu_preparation::apply_tensor_without_format(offset_sizes, ctx.weight_offset.value().options());
    }
}

static void prepare_out_bias(QuantContext &ctx) {
    if (ctx.bias.has_value() && ctx.bias.value().defined()) {
        auto bias_sizes = op_infer::array_to_small_vector(ctx.bias.value().sizes());
        ctx.out_bias = npu_preparation::apply_tensor_without_format(bias_sizes, ctx.bias.value().options());
    }
}

static void execute_prepares(const std::vector<PrepareFunc> &prepares, QuantContext &ctx) {
    for (const auto &prepare : prepares) {
        prepare(ctx);
    }
}

static const std::unordered_map<c10_npu::SocVersion, std::vector<DataFlowConfig>> SOC_DATA_FLOW_CONFIG_MAP = {
    {c10_npu::SocVersion::Ascend950,
        {{judge_mm_mx_a8w4,
             {// FP4 逻辑 C0 为 32，torch 层用 1 个 int8/uint8 元素打包 2 个 FP4，因此物理存储的 C0 维长度为 16
                 prepare_out_weight_nz<false, NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_16>,
                 prepare_out_weight_scale_mx<false>, prepare_out_weight_offset, prepare_out_bias}},
            {judge_gmm_mx_a8w4,
                {// FP4 逻辑 C0 为 32，torch 层用 1 个 int8/uint8 元素打包 2 个 FP4，因此物理存储的 C0 维长度为 16
                    prepare_out_weight_nz<true, NZ_C0_16, ACL_FORMAT_FRACTAL_NZ_C0_16>,
                    prepare_out_weight_scale_mx<true>, prepare_out_weight_offset, prepare_out_bias}}}},
};

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_weight_quant_preprocess(const at::Tensor &weight,
    const at::Tensor &weight_scale, int64_t x_dtype, int64_t weight_dtype, int64_t weight_scale_dtype,
    const c10::optional<at::Tensor> &weight_offset, const c10::optional<at::Tensor> &bias,
    c10::optional<int64_t> x_scale_dtype, c10::optional<int64_t> k_group_size) {
    auto soc_version = c10_npu::GetSocVersion();
    auto it = SOC_DATA_FLOW_CONFIG_MAP.find(soc_version);
    TORCH_CHECK(it != SOC_DATA_FLOW_CONFIG_MAP.end(),
        "Unsupported NPU architecture: " + std::to_string(static_cast<int>(soc_version)) +
            OPS_ERROR(ErrCode::NOT_SUPPORT));

    int64_t k_group_size_real = k_group_size.value_or(0);

    QuantContext ctx{weight, weight_scale, weight_offset, bias, x_dtype, weight_dtype, x_scale_dtype,
        weight_scale_dtype, k_group_size_real, soc_version, false, at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor()};

    bool matched = false;
    for (const auto &[judge, prepares] : it->second) {
        if (judge(ctx)) {
            execute_prepares(prepares, ctx);
            matched = true;
            break;
        }
    }
    TORCH_CHECK(matched, "Unsupported data flow combination." + OPS_ERROR(ErrCode::PARAM));

    TensorWrapper weight_wrapper = make_wrapper(ctx.weight, c10::optional<int64_t>(ctx.weight_dtype));
    TensorWrapper scale_wrapper = make_wrapper(ctx.weight_scale, c10::optional<int64_t>(ctx.weight_scale_dtype));
    TensorWrapper out_weight_wrapper = make_wrapper(ctx.out_weight, c10::optional<int64_t>(ctx.weight_dtype));
    TensorWrapper out_scale_wrapper =
        make_wrapper(ctx.out_weight_scale, c10::optional<int64_t>(ctx.weight_scale_dtype));

    aclDataType x_acl_type = c10_npu::GetAclDataType(x_dtype);
    aclDataType x_scale_acl_type =
        x_scale_dtype.has_value() ? c10_npu::GetAclDataType(x_scale_dtype.value()) : ACL_DT_UNDEFINED;

    at::Tensor weight_offset_tensor = weight_offset.value_or(at::Tensor());
    at::Tensor bias_tensor = bias.value_or(at::Tensor());
    EXEC_NPU_CMD(aclnnWeightQuantPreprocess, weight_wrapper, scale_wrapper, weight_offset_tensor, bias_tensor,
        x_acl_type, x_scale_acl_type, ctx.k_group_size, out_weight_wrapper, out_scale_wrapper, ctx.out_weight_offset,
        ctx.out_bias);

    return std::make_tuple(ctx.out_weight, ctx.out_weight_scale, ctx.out_weight_offset, ctx.out_bias);
}

} // namespace op_api
