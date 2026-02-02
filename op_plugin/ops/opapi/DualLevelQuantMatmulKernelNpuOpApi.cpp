// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
using npu_preparation = at_npu::native::OpPreparation;
constexpr int DIM_ZERO = 0;
constexpr int DIM_ONE = 1;
constexpr int DIM_TWO = 2;
constexpr int SHAPE_SIZE_TWO = 2;
const int64_t B4_NUMS_IN_B8 = 2;

static bool is_nz_format(const at::Tensor &x2)
{
    auto* npu_impl = torch_npu::NPUBridge::GetNpuStorageImpl(x2);
    TORCH_CHECK(npu_impl != nullptr, "Invalid NPU tensor implementation.", OPS_ERROR(ErrCode::PTR));
    const torch_npu::NPUStorageDesc &tensor_desc = npu_impl->npu_desc_;
    return tensor_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ;
}

at::Tensor npu_dual_level_quant_matmul(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &x1_level0_scale,
                                       const at::Tensor &x2_level0_scale, const at::Tensor &x1_level1_scale,
                                       const at::Tensor &x2_level1_scale, const c10::optional<at::Tensor> &bias,
                                       int64_t output_dtype)
{
    TORCH_CHECK(c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend950,
                "This interface is supported only on the Ascend950 platform and after.", OPS_ERROR(ErrCode::PARAM));

    auto x1_dim_num = x1.dim();
    TORCH_CHECK(x1_dim_num == SHAPE_SIZE_TWO, "The shape dimension of x1 is only supported to be 2, but it is ",
                x1_dim_num, OPS_ERROR(ErrCode::PARAM));

    auto x2_dim_num = x2.dim();
    TORCH_CHECK(x2_dim_num == SHAPE_SIZE_TWO, "The shape dimension of x2 is only supported to be 2, but it is ",
                x2_dim_num, OPS_ERROR(ErrCode::PARAM));

    bool is_weight_nz = is_nz_format(x2);
    TORCH_CHECK(is_weight_nz, "x2 only supports NZ format.", OPS_ERROR(ErrCode::PARAM));

    auto x1_k_dim = x1.size(DIM_ONE) * B4_NUMS_IN_B8;  // x1 shape (m, k/2)
    auto x2_k_dim = x2.size(DIM_ONE) * B4_NUMS_IN_B8;  // x2 shape (n, k/2)
    TORCH_CHECK(x1_k_dim == x2_k_dim, "The k of x1 and x2 should be equal. but x1_k_dim is ", x1_k_dim,
                ", x2_k_dim is ", x2_k_dim, OPS_ERROR(ErrCode::PARAM));

    auto out_dim_num = x1_dim_num;
    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[DIM_ZERO] = x1.size(DIM_ZERO);
    output_size[DIM_ONE] = x2.size(DIM_ZERO);

    const at::Tensor &bias_v = bias.value_or(at::Tensor());
    int l0_group_size = 512;
    int l1_group_size = 32;

    TensorWrapper x1_wrapper =
        make_wrapper(x1, c10::optional<int64_t>(static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1)));
    TensorWrapper x1_level1_scale_wrapper =
        make_wrapper(x1_level1_scale, c10::optional<int64_t>(static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0)));
    TensorWrapper x2_level1_scale_wrapper =
        make_wrapper(x2_level1_scale, c10::optional<int64_t>(static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0)));

    c10::TensorOptions options;
    aclDataType output_acltype = c10_npu::GetAclDataType(output_dtype);
    options = x1.options().dtype(npu_preparation::convert_to_scalar_type(output_acltype));
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);

    bool transpose_x1 = false;
    bool transpose_x2 = true;

    EXEC_NPU_CMD(aclnnDualLevelQuantMatmulWeightNz, x1_wrapper, x2, x1_level0_scale, x2_level0_scale,
                 x1_level1_scale_wrapper, x2_level1_scale_wrapper, bias_v, transpose_x1, transpose_x2, l0_group_size,
                 l1_group_size, result);

    return result;
}
}  // namespace op_api
