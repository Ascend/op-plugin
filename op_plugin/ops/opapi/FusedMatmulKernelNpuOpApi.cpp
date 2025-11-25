// Copyright (c) 2025 Huawei Technologies Co., Ltd

// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
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

namespace {
constexpr size_t X_DIM = 2;
constexpr size_t FUSED_TYPE_ARRAY_SIZE = 100;
void infer_out_batch_shape(const at::Tensor &x1, const at::Tensor &x2, std::vector<uint64_t> &batch_record)
{
    auto x1_dim_num = x1.dim();
    auto x2_dim_num = x2.dim();
    auto out_dim_num = std::max(x1_dim_num, x2_dim_num);
    auto &shape_long = x1_dim_num > x2_dim_num ? x1 : x2;
    auto &shape_short = x1_dim_num > x2_dim_num ? x2 : x1;
    int64_t vaild_offset = out_dim_num - std::min(x1_dim_num, x2_dim_num);
    for (int64_t i = 0; i < out_dim_num - X_DIM; i++) {
        auto short_dim = i < vaild_offset ? 1 : shape_short.size(i - vaild_offset);
        auto long_dim = shape_long.size(i);
        TORCH_CHECK(!(short_dim > 1 && long_dim > 1 && short_dim != long_dim),
                    "the x1 shape and x2 shape not supported for broadcast, the short_dim is ", short_dim,
                    " and  the long_dim is ", long_dim, OPS_ERROR(ErrCode::PARAM));
        uint64_t cur_batch_value = static_cast<uint64_t>(std::max(short_dim, long_dim));
        batch_record.push_back(cur_batch_value);
    }
}
} // namespace

at::Tensor npu_fused_matmul(
    const at::Tensor &x, const at::Tensor &x2,
    const c10::optional<at::Tensor> &bias, const c10::optional<at::Tensor> &x3,
    c10::string_view fused_op_type
    )
{
    auto x1_dim_num = x.dim();
    TORCH_CHECK(x1_dim_num >= X_DIM, "x1 shape dim num cannot be less than 2, but it is ",
                x1_dim_num);
    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num >= X_DIM, "x2 shape dim num cannot be less than 2, but it is ",
                x2_dim_num);

    auto x1_m_dim = x.size(x1_dim_num - X_DIM);
    auto x1_k_dim = x.size(x1_dim_num - 1);
    auto x2_n_dim = x2.size(x2_dim_num - 1);
    auto x2_k_dim = x2.size(x2_dim_num - X_DIM);
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ",
                x1_k_dim, ", x2_k_dim is ", x2_k_dim);

    std::vector<uint64_t> batch_record;
    infer_out_batch_shape(x, x2, batch_record);
    const at::Tensor long_tensor = x1_dim_num > x2_dim_num ? x : x2;
    auto output_size = op_infer::array_to_small_vector(long_tensor.sizes());
    output_size[long_tensor.dim() - X_DIM] = x1_m_dim;
    output_size[long_tensor.dim() - 1] = x2_n_dim;
    for (int64_t i = 0; i < long_tensor.dim() - X_DIM; i++) {
        output_size[i] = static_cast<int64_t>(batch_record[i]);
    }
      
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                             x.dtype());
    const at::Tensor &x3_real = x3.value_or(at::Tensor());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());

    int8_t cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowMatmulHF32());
    char fused_type[FUSED_TYPE_ARRAY_SIZE] = {0};
    TORCH_CHECK(std::string(fused_op_type).size() <= FUSED_TYPE_ARRAY_SIZE,
                "the len of fused_op_type is bigger than the default");
    std::string(fused_op_type).copy(fused_type, FUSED_TYPE_ARRAY_SIZE);
    EXEC_NPU_CMD(aclnnFusedMatmul, x, x2, bias_real, x3_real, fused_type, cube_math_type, result);
    return result;
}
}