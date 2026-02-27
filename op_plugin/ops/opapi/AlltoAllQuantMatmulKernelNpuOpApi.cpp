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

static const int TWO_DIMS = 2;
static const int DYN_PERTOKEN_QUANT_MODE = 7;
static const int64_t PERCHANNEL_QUANT_MODE = 2;
static const int64_t NON_QUANT = 0;
static const int64_t NON_GROUP = 0;
static const int64_t ACL_UNDEFINED = -1;
static const int64_t ACL_FP8_E5M2 = 35;

// world_size
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16};


std::tuple<at::Tensor, at::Tensor> npu_all_to_all_quant_matmul(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom,
    int64_t world_size, bool all2all_out_flag, const c10::optional<at::Tensor>& bias, const c10::optional<at::Tensor>& x1_scale,
    const c10::optional<at::Tensor>& x2_scale, const c10::optional<at::Tensor>& common_scale,
    const c10::optional<at::Tensor>& x1_offset, const c10::optional<at::Tensor>& x2_offset,
    c10::optional<int64_t> x1_quant_mode, c10::optional<int64_t> x2_quant_mode, c10::optional<int64_t> common_quant_mode,
    c10::OptionalIntArrayRef group_sizes, c10::OptionalIntArrayRef all2all_axes,
    c10::optional<int64_t> comm_quant_dtype, c10::optional<int64_t> x1_quant_dtype,
    c10::optional<int64_t> x1_dtype, c10::optional<int64_t> x2_dtype,
    c10::optional<int64_t> x1_scale_dtype, c10::optional<int64_t> x2_scale_dtype,
    c10::optional<int64_t> output_scale_dtype, c10::optional<int64_t> comm_scale_dtype, c10::optional<int64_t> y_dtype)
{
    // 校验x1x2的shape, 2维(m, k) or (k, n)
    TORCH_CHECK(x1.dim() == TWO_DIMS, "The x1 input of alltoallquantmatmul is required to be 2D, but the actual x1 input is ", x1.dim(), "D." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() == TWO_DIMS, "The x2 input of alltoallquantmatmul is required to be 2D, but the actual x2 input is ", x2.dim(), "D." + OPS_ERROR(ErrCode::PARAM));

    // 校验world_size
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
        "The world_size should be in [2, 4, 8, 16], but the actual value is ", world_size, "." + OPS_ERROR(ErrCode::VALUE));
    if (world_size != 0) {
        TORCH_CHECK(x1.size(0) % world_size == 0, "The x1 first-axis should be divisible by world_size.", OPS_ERROR(ErrCode::PARAM));
    }

    // 处理group_sizes
    at::IntArrayRef group_size_list = group_sizes.value_or(at::IntArrayRef{});
    int64_t group_size = op_plugin::utils::check_and_get_group_size(group_size_list);

    // pta主要是为了推导output的shape和dtype，如果这里的y_dtype没有传入，则默认是fp32
    int64_t output_default_dtype = (y_dtype.has_value() && y_dtype.value() != ACL_UNDEFINED) ? y_dtype.value() : static_cast<int64_t>(at::ScalarType::Float);
    aclDataType output_acl_type = c10_npu::GetAclDataType(output_default_dtype);
    at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);
    // 推导输出output_tensor的shape
    // 不允许转置，因此可以确定直接确定m和n的值
    int64_t out_m = x1.size(0);
    if (world_size != 0) {
        out_m = x1.size(0) / world_size;
    }
    int64_t out_n = x2.size(1);
    auto output_size = {out_m, out_n};
    // output_tensor按照实际的shape和dtype去创建
    at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(output_scalar_type));

    // 生成aclnn接口需要的默认值
    char *group_ptr = const_cast<char *>(hcom.data());
    int64_t x1QuantMode = x1_quant_mode.has_value() ? x1_quant_mode.value() : DYN_PERTOKEN_QUANT_MODE;
    int64_t x2QuantMode = x2_quant_mode.has_value() ? x2_quant_mode.value() : PERCHANNEL_QUANT_MODE;
    int64_t commonQuantMode = common_quant_mode.has_value() ? common_quant_mode.value() : NON_QUANT;
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    int64_t commQuantDtype = comm_quant_dtype.has_value() ? comm_quant_dtype.value() : ACL_UNDEFINED;
    int64_t x1QuantDtype = x1_quant_dtype.has_value() ? x1_quant_dtype.value() : ACL_FP8_E5M2;

    // mx量化下scale为fp8_e8m0，需要wrapper包装
    const at::Tensor &x1_scale_real = x1_scale.value_or(at::Tensor());
    const at::Tensor &x2_scale_real = x2_scale.value_or(at::Tensor());
    TensorWrapper x1_scale_wrapper = make_wrapper(x1_scale_real, x1_scale_dtype);
    TensorWrapper x2_scale_wrapper = make_wrapper(x2_scale_real, x2_scale_dtype);

    // 推导alltoallOutput
    if (all2all_out_flag) {
        aclDataType alltoall_out_acl_type = npu_preparation::convert_to_acl_data_type(x1.scalar_type());
        at::ScalarType alltoall_out_scalar_type = npu_preparation::convert_to_scalar_type(alltoall_out_acl_type);
        auto alltoall_out_size = {out_m, x1.size(1) * world_size};
        at::Tensor alltoall_out_tensor = npu_preparation::apply_tensor_without_format(alltoall_out_size, c10::dtype(alltoall_out_scalar_type));
        // 调用aclnn接口
        EXEC_NPU_CMD(aclnnAlltoAllQuantMatmul, x1, x2, bias, x1_scale_wrapper, x2_scale_wrapper, common_scale, x1_offset, x2_offset, group_ptr, all2all_axes,
            x1QuantMode, x2QuantMode, commonQuantMode, commQuantDtype, x1QuantDtype, group_size, transpose_x1, transpose_x2,
            output_tensor, alltoall_out_tensor);
        return std::tuple<at::Tensor, at::Tensor>(output_tensor, alltoall_out_tensor);
    } else {
        const std::nullptr_t& alltoalloutNullptr = nullptr;
        // 调用aclnn接口
        EXEC_NPU_CMD(aclnnAlltoAllQuantMatmul, x1, x2, bias, x1_scale_wrapper, x2_scale_wrapper, common_scale, x1_offset, x2_offset, group_ptr, all2all_axes,
            x1QuantMode, x2QuantMode, commonQuantMode, commQuantDtype, x1QuantDtype, group_size, transpose_x1, transpose_x2,
            output_tensor, alltoalloutNullptr);
        return std::tuple<at::Tensor, at::Tensor>(output_tensor, alltoalloutNullptr);
    }
}
} // namespace op_api