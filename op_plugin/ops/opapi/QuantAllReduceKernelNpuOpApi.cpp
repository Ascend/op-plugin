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
#include "op_plugin/utils/KernelNpuOutputSize.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
static const int NUM_64 = 64;
static const int NUM_128 = 128;
static const int H_LOWER_LIMIT = 1024;
static const int H_UPPER_LIMIT = 8192;

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
// scales valid dtype
const std::set<int64_t> SUPPORT_SCALES_DTYPE_LIST{
    static_cast<int64_t>(c10_npu::DType::FLOAT),
    static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0)
};

static c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }
    return shape_small_vec;
}

at::Tensor npu_quant_all_reduce(const at::Tensor &x, const at::Tensor &scales, c10::string_view hcom,
                                int64_t world_size, c10::optional<c10::string_view> reduce_op,
                                c10::optional<int64_t> output_dtype, c10::optional<int64_t> x_dtype,
                                c10::optional<int64_t> scales_dtype)
{
    // 校验x的shape, 2维(bs, h)或3维(b, s, h)
    TORCH_CHECK(x.dim() == DIM_TWO || x.dim() == DIM_THREE, "The input of mm is required to be 2D, but the actual input is ",
                x.dim(), OPS_ERROR(ErrCode::PARAM));
    // 校验x的dtype
    if (x_dtype.has_value()) {
        TORCH_CHECK(SUPPORT_X_DTYPE_LIST.find(x_dtype.value()) != SUPPORT_X_DTYPE_LIST.end(),
                    "The optional parameter x_dtype only supports int8/hifloat8/float8_e4m3fn/float8_e5m2, but now is ",
                    op_plugin::utils::DTypeToString(x_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    // 校验world_size, 这一行代码仅为了过门禁而做的非0校验
    TORCH_CHECK(world_size != 0, "The world_size can not be zero", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
                "The world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, OPS_ERROR(ErrCode::VALUE));

    // x.shape是(bs, h)或者(b, s, h)，所以第0维可能是bs，也可能是b
    int64_t axis_bs = x.size(0);
    if (x.dim() == DIM_THREE) {
        axis_bs = axis_bs * x.size(1);
    }
    TORCH_CHECK(axis_bs % world_size == 0, "The x BS-axis should be divisible by world_size",
                OPS_ERROR(ErrCode::PARAM));

    // (bs, h)或者(b, s, h), h范围在[1024, 8192]内，且h满足128对齐
    uint32_t axis_h = (x.dim() == DIM_THREE ? 2 : 1);
    TORCH_CHECK(x.size(axis_h) >= H_LOWER_LIMIT && x.size(axis_h) <= H_UPPER_LIMIT && x.size(axis_h) % NUM_128 == 0,
                "The x H-axis should be in [1024, 8192] and divisible by 128", OPS_ERROR(ErrCode::PARAM));

    // 校验scales的dtype
    if (scales_dtype.has_value()) {
        TORCH_CHECK(SUPPORT_SCALES_DTYPE_LIST.find(scales_dtype.value()) != SUPPORT_SCALES_DTYPE_LIST.end(),
                    "The optional parameter scales_dtype only supports float/float_e8m0, but now is ",
                    op_plugin::utils::DTypeToString(scales_dtype.value()), "." + OPS_ERROR(ErrCode::VALUE));
    }

    // pta主要是为了推导output的shape和dtype，如果这里的output_dtype没有传入，则默认是bf16
    int64_t output_default_dtype = static_cast<int64_t>(at::ScalarType::BFloat16);
    if (output_dtype.has_value()) {
        // 这里应该校验output_dtype，但是目前没有bfloat16的类型定义。避免影响正常功能，因此这里不校验了
        output_default_dtype = output_dtype.value();
    }
    aclDataType output_acl_type = c10_npu::GetAclDataType(output_default_dtype);
    at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);
    // 推导output_tensor：output_tensor按照实际的shape和dtype去创建，output的shape和传入的x的shape完全一致
    at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(array_to_small_vector(x.sizes()), c10::dtype(output_scalar_type));

    // attr
    char *group_ptr = const_cast<char *>(hcom.data());
    c10::string_view reduce_op_value = reduce_op.value_or("sum");
    char *reduce_op_ptr = const_cast<char *>(reduce_op_value.data());

    // 海思自定义dtype，使用wrapper封装，相当于打一个标签，标记真实的属性，让aclnn接口识别到传入tensor具体的dtype
    TensorWrapper x_wrapper = {x, (x_dtype.has_value()) ?
                               c10_npu::GetAclDataType(x_dtype.value()) :
                               npu_preparation::convert_to_acl_data_type(x.scalar_type())};
    TensorWrapper scales_wrapper = {scales, (scales_dtype.has_value()) ?
                                    c10_npu::GetAclDataType(scales_dtype.value()) :
                                    npu_preparation::convert_to_acl_data_type(scales.scalar_type())};

    // 前面的wrapper打包传进去之后，这里直接调用aclnn接口
    EXEC_NPU_CMD(aclnnQuantAllReduce, x_wrapper, scales_wrapper, group_ptr, reduce_op_ptr, output_tensor);
    return output_tensor;
}

} // namespace op_api
