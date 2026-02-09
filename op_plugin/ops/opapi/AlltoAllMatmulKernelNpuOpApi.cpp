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

// world_size
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16};

std::tuple<at::Tensor, at::Tensor> npu_all_to_all_matmul(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom, int64_t world_size,
                                                         const c10::optional<at::Tensor>& bias, c10::OptionalIntArrayRef all2all_axes, bool all2all_out_flag)
{
    // 校验x的shape, 2维(m, k) or (k, n)
    TORCH_CHECK(x1.dim() == TWO_DIMS, "The x1 input of alltoallmatmul is required to be 2D, but the actual x1 input is ", x1.dim(), "D." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() == TWO_DIMS, "The x2 input of alltoallmatmul is required to be 2D, but the actual x2 input is ", x2.dim(), "D." + OPS_ERROR(ErrCode::PARAM));

    // 校验world_size
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
        "The world_size should be in [2, 4, 8, 16], but the actual value is ", world_size, "." + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x1.size(0) % world_size == 0, "The x1 first-axis should be divisible by world_size.", OPS_ERROR(ErrCode::PARAM));

    // pta主要是为了推导output的shape和dtype，非量化matmulalltoall的output_dtype和输入x1一致
    aclDataType output_acl_type = npu_preparation::convert_to_acl_data_type(x1.scalar_type());
    at::ScalarType output_scalar_type = npu_preparation::convert_to_scalar_type(output_acl_type);

    // 推导输出output_tensor的shape
    // 不允许转置，因此可以确定直接确定m和n的值
    int64_t out_m = x1.size(0) / world_size;
    int64_t out_n = x2.size(1);
    auto output_size = {out_m, out_n};
    // output_tensor按照实际的shape和dtype去创建
    at::Tensor output_tensor = npu_preparation::apply_tensor_without_format(output_size, c10::dtype(output_scalar_type));

    // 生成aclnn接口需要的默认值
    char *group_ptr = const_cast<char *>(hcom.data());
    bool transpose_x1 = false;
    bool transpose_x2 = false;

    // 推导alltoallOutput
    if (all2all_out_flag) {
        aclDataType alltoall_out_acl_type = npu_preparation::convert_to_acl_data_type(x1.scalar_type());
        at::ScalarType alltoall_out_scalar_type = npu_preparation::convert_to_scalar_type(alltoall_out_acl_type);
        auto alltoall_out_size = {out_m, x1.size(1) * world_size};
        at::Tensor alltoall_out_tensor = npu_preparation::apply_tensor_without_format(alltoall_out_size, c10::dtype(alltoall_out_scalar_type));
        // 调用aclnn接口
        EXEC_NPU_CMD(aclnnAlltoAllMatmul, x1, x2, bias, all2all_axes, group_ptr, transpose_x1, transpose_x2, output_tensor, alltoall_out_tensor);
        return std::tuple<at::Tensor, at::Tensor>(output_tensor, alltoall_out_tensor);
    } else {
        const std::nullptr_t& alltoalloutNullptr = nullptr;
        // 调用aclnn接口
        EXEC_NPU_CMD(aclnnAlltoAllMatmul, x1, x2, bias, all2all_axes, group_ptr, transpose_x1, transpose_x2, output_tensor, alltoalloutNullptr);
        return std::tuple<at::Tensor, at::Tensor>(output_tensor, alltoalloutNullptr);
    }
}
} // namespace op_api