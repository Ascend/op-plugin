// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

void check_params(const at::Tensor &x1, const at::Tensor &x2,
                  const c10::optional<at::Tensor> &antiquant_scale,
                  const c10::optional<at::Tensor> &antiquant_offset,
                  const c10::optional<at::Tensor> &x3,
                  const c10::optional<at::Tensor> &dequant_scale)
{
    // 对x1和x2的shape进行校验，满足matmul计算规则
    TORCH_CHECK(x2.dim() == 2, "x2 needs to be 2D, but got: ", x2.dim(), "D");
    TORCH_CHECK(x1.size(x1.dim() - 1) == x2.size(0), "K of x1 and x2 should be same, but they are x1_k: ",
                x1.size(x1.dim() - 1), ", x2_k: ", x2.size(0));

    // 由于aclnn接口不接收冗余传入的参数，aclnn内部无法提示冗余参数，需要在此进行校验。
    // 对于MC2非量化场景，aclnn接口不对antiquant_scale、antiquant_offset、dequant_scale进行处理，在此校验住，不要传入冗余参数。
    // A8W8; antiquantScale NULL antiquantOffset NULL
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x1.scalar_type() == at::kChar, "x1 must be an int8 tensor for quant.");
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for quant.");
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value()),
                    "when both dtype of x1 and dtype of x2 are equal to int8, "
                    "antiquantScale, antiquantOffset should both be null");
    }
    // 对于MC2伪量化场景A16W8，aclnn接口不对dequant_scale进行处理，在此校验住，不要传入冗余参数。
    // A16W8; dequantScale NULL
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for weight quant.");
        TORCH_CHECK((!dequant_scale.has_value()),
                    "when only dtype of x2 is equal to int8, dequantScale should be null");
    }
    // 对于MC2全量化场景A8W8，aclnn接口不对antiquant_scale、antiquant_offset进行处理，在此校验住，不要传入冗余参数。
    // MC2 without quant; antiquantScale NULL antiquantOffset NULL dequantScale NULL
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value() && !dequant_scale.has_value()),
                    "when neither dtype of x1 or dtype of x2 is equal to int8, "
                    "antiquantScale, antiquantOffset and dequantScale should all be null");
    }

    // check x3 dtype and shape
    if (x3.has_value()) {
        auto output_size = op_infer::array_to_small_vector(x1.sizes());
        output_size[x1.dim() - 1] = x2.size(1);
        // a8w8: dtype of output should be half.
        auto output_dtype = x1.scalar_type() == at::kChar ? at::ScalarType::Half : x1.scalar_type();
        const at::Tensor &x3_real = x3.value_or(at::Tensor());
        TORCH_CHECK(x3_real.sizes().equals(output_size), "x3 with shape ", x3_real.sizes(),
                    " doesn't match the output shape ", output_size);
        TORCH_CHECK(x3_real.scalar_type() == output_dtype, "x3 with dtype ", x3_real.scalar_type(),
                    " doesn't match the output dtype ", output_dtype);
    }
}

at::Tensor npu_mm_all_reduce_base(const at::Tensor &x1, const at::Tensor &x2, c10::string_view hcom,
                                  c10::string_view reduce_op, const c10::optional<at::Tensor> &bias,
                                  const c10::optional<at::Tensor> &antiquant_scale,
                                  const c10::optional<at::Tensor> &antiquant_offset,
                                  const c10::optional<at::Tensor> &x3, const c10::optional<at::Tensor> &dequant_scale,
                                  int64_t antiquant_group_size, int64_t comm_turn)
{
    check_params(x1, x2, antiquant_scale, antiquant_offset, x3, dequant_scale);
    // size of last dim of output should be the same as size of last dim of x2
    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[x1.dim() - 1] = x2.size(1);
    // a8w8: dtype of output should be half.
    auto output_dtype = x1.scalar_type() == at::kChar ? at::ScalarType::Half : x1.scalar_type();
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                             x1.options().dtype(output_dtype));
    char *reduce_op_ptr = const_cast<char *>(reduce_op.data());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    const at::Tensor &x3_real = x3.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // a8w8: x1\x2 kChar; a16w8: x2 kChar;
    if (!isIntegralType(x2.scalar_type())) {
        if (x3.has_value()) {
            EXEC_NPU_CMD(aclnnMatmulAllReduceV2, x1, x2, bias_real, x3_real, hcom_ptr, reduce_op_ptr, comm_turn,
                         stream_mode, x3_real);
            return x3_real;
        } else {
            EXEC_NPU_CMD(aclnnMatmulAllReduce, x1, x2, bias_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode,
                         result);
        }
    } else if (isIntegralType(x1.scalar_type())) {
        const at::Tensor &dequant_scale_real = dequant_scale.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnQuantMatmulAllReduce, x1, x2, bias_real, x3_real, dequant_scale_real, hcom_ptr,
                     reduce_op_ptr, comm_turn, stream_mode, result);
    } else {
        const at::Tensor &antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor &antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        if (x3.has_value()) {
            EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_real, antiquant_scale_real, antiquant_offset_real,
                         x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, antiquant_group_size, x3_real);
            return x3_real;
        } else {
            EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_real, antiquant_scale_real, antiquant_offset_real,
                         x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, antiquant_group_size, result);
        }
    }

    return result;
}
}  // namespace op_api
