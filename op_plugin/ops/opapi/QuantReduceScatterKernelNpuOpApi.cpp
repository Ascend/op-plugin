// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <set>
#include <op_plugin/OpApiInterface.h>
#include <torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h>
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// world_size
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16, 32};
// x valid dtype
const std::set<int64_t> SUPPORT_X_DTYPE_LIST{
    static_cast<int64_t>(c10_npu::DType::INT8),
    static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
    static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2),
    static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN),
    static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2),
    static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)
};

static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
static const int NUM_64 = 64;
static const int NUM_128 = 128;

at::Tensor npu_quant_reduce_scatter(const at::Tensor &x, const at::Tensor &scales, c10::string_view hcom,
                                    int64_t world_size, c10::optional<c10::string_view> reduce_op,
                                    c10::optional<int64_t> output_dtype, c10::optional<int64_t> x_dtype,
                                    c10::optional<int64_t> scales_dtype)
{
    // 校验x的shape, 2维(bs, h)
    TORCH_CHECK(x.dim() == DIM_TWO, "The x shape is required to be 2 dim, but the actual shape is ", x.dim(),
                OPS_ERROR(ErrCode::PARAM));
    // 校验x是否为空tensor
    TORCH_CHECK(x.size(0) != 0 && x.size(1) != 0, "The input 2 dim tensor x can not be empty tensor", OPS_ERROR(ErrCode::PARAM));
    // 校验x的dtype: int8/hifloat8/float8_e5m2/float8_e4m3fn/float4_e1m2/float4_e2m1
    if (x_dtype.has_value()) {
        TORCH_CHECK(SUPPORT_X_DTYPE_LIST.find(x_dtype.value()) != SUPPORT_X_DTYPE_LIST.end(),
                    "The optional parameter x_dtype only supports int8/hifloat8/float8_e4m3fn/float8_e5m2, but now is ",
                    op_plugin::utils::DTypeToString(x_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
                "The world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x.size(0) % world_size == 0, "The x BS-axis should be divisible by world_size",
                OPS_ERROR(ErrCode::PARAM));
    // (bs, h), h满足128对齐
    TORCH_CHECK(x.size(1) % NUM_128 == 0, "The x H-axis should be divisible by 128",
                OPS_ERROR(ErrCode::PARAM));

    // 校验scales的shape
    TORCH_CHECK(scales.dim() == DIM_TWO || scales.dim() == DIM_THREE,
                "The input scales tensor shape is required to be equal to x in TG QuantMode, "
                "or be equal to x plus 1 in MX QuantMode, but the actual input scales shape is ",
                scales.dim(), OPS_ERROR(ErrCode::PARAM));
    // 校验scales是否为空tensor
    if (scales.dim() == DIM_TWO) {
        TORCH_CHECK(scales.size(0) != 0 && scales.size(1) != 0, "The input 2 dim tensor scales can not be empty tensor",
                    OPS_ERROR(ErrCode::PARAM));
    } else if (scales.dim() == DIM_THREE) {
        TORCH_CHECK(scales.size(0) != 0 && scales.size(1) != 0 && scales.size(DIM_TWO) != 0,
                    "The input 3 dim tensor scales can not be empty tensor", OPS_ERROR(ErrCode::PARAM));
    }
    // 校验scales的dtype: float/float8_e8m0
    if (scales_dtype.has_value()) {
        TORCH_CHECK(scales_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT) ||
                    scales_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                    "The optional parameter scales_dtype only supports float/float_e8m0, but now is ",
                    op_plugin::utils::DTypeToString(scales_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    // pta主要是为了推导output的shape和dtype，如果这里的output_dtype没有传入，则默认是bf16
    int64_t output_default_dtype = static_cast<int64_t>(at::ScalarType::BFloat16);
    if (output_dtype.has_value()) {
        // 这里应该校验output_dtype，但是目前没有bfloat16的类型定义。怕影响正常功能，因此这里不校验了
        output_default_dtype = output_dtype.value();
    }
    aclDataType output_acl_type = c10_npu::GetAclDataType(output_default_dtype);
    at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);
    // 输出的output_tensor需要自己推导，这里对最终结果做了一个reduce scatter，所以m轴应该是输入x的m轴的1/rank
    auto output_size = {x.size(0) / world_size, x.size(1)};
    // output_tensor按照实际的shape和dtype去创建
    at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(output_scalar_type));

    // attr
    char *group_ptr = const_cast<char *>(hcom.data());
    c10::string_view reduce_op_value = reduce_op.value_or("sum");
    char *reduce_op_ptr = const_cast<char *>(reduce_op_value.data());

    // 如果是自定义的dtype的话，那么这里就需要使用一个wrapper
    // 让aclnn接口识别到传入的tensor具体的dtype类型，相当于打一个标签，标记真实的属性
    TensorWrapper x_wrapper = make_wrapper(x, x_dtype);
    TensorWrapper scales_wrapper = make_wrapper(scales, scales_dtype);

    // 前面的wrapper打包传进去之后，这里直接调用aclnn接口
    EXEC_NPU_CMD(aclnnQuantReduceScatter, x_wrapper, scales_wrapper, group_ptr, reduce_op_ptr, output_tensor);
    return output_tensor;
}

} // namespace op_api
