// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

const int64_t INT4_NUMS_IN_INT32_SPACE = 8;

at::Tensor npu_group_quant(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& group_index,
    const c10::optional<at::Tensor>& offset,
    c10::optional<at::ScalarType> dst_dtype)
{
    at::ScalarType dst_type = c10::value_or_else(dst_dtype, [] {return at::ScalarType::Char;});
    if (dst_type == at::kQInt8) {
        dst_type = at::kChar;
    }

    TORCH_CHECK(dst_type == at::ScalarType::Char || dst_type == at::ScalarType::QUInt4x2,
                "dst_dtype must be Int8 or Int4" + OPS_ERROR(ErrCode::TYPE));
    at::Tensor result;
    if (dst_type == at::ScalarType::QUInt4x2) {
        auto output_shape = op_infer::array_to_small_vector(x.sizes());
        auto x_dim_num = x.dim();
        TORCH_CHECK(output_shape[x_dim_num - 1] % INT4_NUMS_IN_INT32_SPACE == 0,
                    "input shape last dim must be divded by 8" + OPS_ERROR(ErrCode::PARAM));
        output_shape[x_dim_num - 1] /= INT4_NUMS_IN_INT32_SPACE;
        dst_type = at::ScalarType::Int;
        result = npu_preparation::apply_tensor_without_format(output_shape, x.options().dtype(dst_type));
    } else {
        result = npu_preparation::apply_tensor_without_format(x.sizes(), x.options().dtype(dst_type));
    }

    EXEC_NPU_CMD(aclnnGroupQuant, x, scale, group_index, offset, dst_type, result);
    return result;
}
}
