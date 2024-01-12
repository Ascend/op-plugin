// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

at::Tensor npu_mm_all_reduce_base(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom,
                                  c10::string_view reduce_op, const c10::optional<at::Tensor> &bias,
                                  const c10::optional<at::Tensor> &antiquant_scale,
                                  const c10::optional<at::Tensor> &antiquant_offset,
                                  const c10::optional<at::Tensor> &x3, int64_t comm_turn)
{
    TORCH_CHECK(x2.dim() == 2, "x2 needs to be 2D, but got: ", x2.dim(), "D");
    // size of last dim of output should be the same as size of last dim of x2
    TORCH_CHECK(x1.size(x1.dim() - 1) == x2.size(0), "K of x1 and x2 should be same, but they are x1_k: ",
                x1.size(x1.dim() - 1), ", x2_k: ", x2.size(0));
    // size of last dim of output should be the same as size of last dim of x2
    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[x1.dim() - 1] = x2.size(1);
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, x1.options());
    char *reduce_op_ptr = const_cast<char *>(reduce_op.data());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &x3_real = x3.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // atomic matmul with add
    if (x3.has_value()) {
        TORCH_CHECK(x3_real.sizes().equals(output_size), "x3 with shape ", x3_real.sizes(),
                    " doesn't match the output shape ", output_size);
        TORCH_CHECK(x3_real.scalar_type() == result.scalar_type(), "x3 with dtype ", x3_real.scalar_type(),
                    " doesn't match the output dtype ", result.scalar_type());
    }

    if (!isIntegralType(x2.scalar_type())) {
        EXEC_NPU_CMD(aclnnMatmulAllReduce, x1, x2, bias_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, result);
    } else {
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for quant.");
        const at::Tensor &antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        if (x3.has_value()) {
            EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_real, antiquant_scale_real, antiquant_offset_real,
                         x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, x3_real);
            return x3_real;
        } else {
            EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_real, antiquant_scale_real, antiquant_offset_real,
                         x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, result);
        }
    }

    return result;
}
}  // namespace op_api
