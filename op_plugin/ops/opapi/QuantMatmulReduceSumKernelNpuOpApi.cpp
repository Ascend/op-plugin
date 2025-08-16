// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
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
constexpr int X1_DIM_NUM = 3;
constexpr int X1_SCALE_DIM = 2;
constexpr int X_LAST_DIM_INDEX = 2;
constexpr int Y_DIM_NUM = 2;
using npu_preparation = at_npu::native::OpPreparation;

static bool is_nz_format(const at::Tensor &w)
{
    const torch_npu::NPUStorageDesc &tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(w)->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

at::Tensor npu_quant_matmul_reduce_sum(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const c10::optional<at::Tensor> &x1_scale_optional,
    const c10::optional<at::Tensor> &x2_scale_optional)
{
    bool is_x2_nz = is_nz_format(x2);
    TORCH_CHECK(is_x2_nz, "only support x2's format is nz.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1.dim() == X1_DIM_NUM, "x1 dim should be ", X1_DIM_NUM, OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2.dim() == X1_DIM_NUM, "x2 dim should be ", X1_DIM_NUM, OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &x1_scale = x1_scale_optional.value_or(at::Tensor());
    TORCH_CHECK(x1_scale.dim() == X1_SCALE_DIM, "x1_scale dim should be ", X1_SCALE_DIM, OPS_ERROR(ErrCode::PARAM));

    const at::Tensor &x2_scale = x2_scale_optional.value_or(at::Tensor());
    TORCH_CHECK(x2_scale.dim() == 1, "x2_scale dim should be ", 1, OPS_ERROR(ErrCode::PARAM));

    auto b_dim = x1.size(0);
    auto m_dim = x1.size(1);
    auto n_dim = x2.size(X_LAST_DIM_INDEX);
    TORCH_CHECK(b_dim == x2.size(0), "the first dim of x1 and x2 must be same", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        x1.size(X_LAST_DIM_INDEX) == x2.size(1), "the K dim of x1 and x2 must be same", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x2_scale.size(0) == n_dim, "the shape of x2_scale must be (N,)", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(x1_scale.size(1) == m_dim && x1_scale.size(0) == b_dim,
        "the shape of x1_scale must be (B, M)",
        OPS_ERROR(ErrCode::PARAM));

    c10::SmallVector<int64_t, Y_DIM_NUM> output_size;
    output_size.push_back(m_dim);
    output_size.push_back(n_dim);
    c10::TensorOptions options = x1.options().dtype(at::ScalarType::BFloat16);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    at::Tensor y_scale;
    at::Tensor x1_offset;
    at::Tensor x2_offset;
    at::Tensor y_offset;
    at::Tensor bias;
    bool transpose_x1 = false;
    bool transpose_x2 = false;
    int64_t group_size = -1;
    c10::IntArrayRef dims = {0};
    bool keep_dims = false;
    EXEC_NPU_CMD(aclnnQuantMatmulReduceSumWeightNz,
                 x1, x2, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, bias,
                 transpose_x1, transpose_x2, group_size, dims, keep_dims, result);
    return result;
}

}  // namespace op_api
