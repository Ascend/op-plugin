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

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
constexpr int64_t QUANT_MODE_2 = 2;
constexpr int64_t QUANT_MODE_3 = 3;
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_swiglu_group_quant(const at::Tensor &x,
    const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &group_index,
    const c10::optional<at::Tensor> &scale, int64_t dst_type, int64_t quant_mode,
    int64_t block_size, bool round_scale, double clamp_limit, double dst_type_max, bool output_origin)
{
    at::Tensor y;
    at::Tensor y_scale;
    at::Tensor y_origin = at::empty({0}, x.options());;

    int64_t x_last_dim = x.size(x.dim() - 1);
    TORCH_CHECK(x_last_dim % 2 == 0, "x last dim size should be even", OPS_ERROR(ErrCode::PARAM));

    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    y_shape[x.dim() - 1] = x_last_dim / 2;

    aclDataType y_acltype;
    if (quant_mode == QUANT_MODE_2 || quant_mode == QUANT_MODE_3) {
        y_acltype = aclDataType::ACL_HIFLOAT8;
        y = npu_preparation::apply_tensor_without_format(y_shape, at::ScalarType::Byte);
    }

    if (quant_mode == QUANT_MODE_2) {
        y_scale = at::empty({0}, x.options().dtype(at::kFloat));
    } else if (quant_mode == QUANT_MODE_3) {
        if (group_index.has_value() && group_index->defined()) {
            y_scale  = npu_preparation::apply_tensor_without_format(group_index.value().sizes(), c10::dtype(at::ScalarType::Float));
        } else {
            y_scale  = npu_preparation::apply_tensor_without_format({1}, c10::dtype(at::ScalarType::Float));
        }
    }

    if (output_origin) {
        auto y_origin_shape = op_infer::array_to_small_vector(x.sizes());
        y_origin_shape[x.dim() - 1] = x_last_dim / 2;
        y_origin = npu_preparation::apply_tensor_without_format(y_origin_shape, x.options());
    }

    TensorWrapper y_wrapper = {y, y_acltype};

    aclDataType y_scale_acltype = npu_preparation::convert_to_acl_data_type(y_scale.scalar_type());

    TensorWrapper y_scale_wrapper = {y_scale, y_scale_acltype};

    TensorWrapper y_origin_wrapper = {y_origin, npu_preparation::convert_to_acl_data_type(y_origin.scalar_type())};

    EXEC_NPU_CMD(aclnnSwigluGroupQuant, x, weight, group_index, scale, y_acltype, quant_mode,
        block_size, round_scale, clamp_limit, dst_type_max, output_origin, y_wrapper, y_scale_wrapper, y_origin_wrapper);

    return std::make_tuple(y, y_scale, y_origin);
}
}  // namespace op_api
