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

at::ScalarType get_output_dtype(const at::Tensor& x1, const c10::optional<at::Tensor>& dequant_scale)
{
    auto output_dtype = x1.scalar_type() == at::kChar ? at::ScalarType::Half : x1.scalar_type();
    if (dequant_scale.has_value()) {
        const at::Tensor& dequant = dequant_scale.value();
        if (dequant.scalar_type() == at::kBFloat16) {
            output_dtype = at::kBFloat16;
        }
    }
    return output_dtype;
}

void check_params(const at::Tensor& x1, const at::Tensor& x2,
                  const c10::optional<at::Tensor>& antiquant_scale,
                  const c10::optional<at::Tensor>& antiquant_offset,
                  const c10::optional<at::Tensor>& x3,
                  const c10::optional<at::Tensor>& dequant_scale,
                  const c10::optional<at::Tensor>& pertoken_scale,
                  const c10::optional<at::Tensor>& comm_quant_scale_1,
                  const c10::optional<at::Tensor>& comm_quant_scale_2)
{
    // check shape: shape of x1:[s,m,k], shape of x2:[k,n], k_x1 == k_x2
    TORCH_CHECK(x2.dim() == 2, "x2 needs to be 2D, but got: ", x2.dim(), "D", OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(x1.size(x1.dim() - 1) == x2.size(0), "K of x1 and x2 should be same, but they are x1_k: ",
                x1.size(x1.dim() - 1), ", x2_k: ", x2.size(0), OPS_ERROR(ErrCode::VALUE));

    // check parameters.
    // aclnn apis for MC2 share one torch_npu api, therefore, each aclnn api only accepts parameters
    // that will be used. Any unused parameter will be seen as illegal. The job must be done here in
    // torch_npu api.
    // A8W8: antiquantScale and antiquantOffset should be None.
    // A16W8: dequantScale should be None.
    // MC2 without quantization. antiquantScale and antiquantOffset and dequantScale should be None.
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x1.scalar_type() == at::kChar, "x1 must be an int8 tensor for quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value()),
                    "when both dtype of x1 and dtype of x2 are equal to int8, "
                    "antiquantScale, antiquantOffset should both be null", OPS_ERROR(ErrCode::TYPE));
    } else if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        TORCH_CHECK(x2.scalar_type() == at::kChar, "x2 must be an int8 tensor for weight quant.", OPS_ERROR(ErrCode::TYPE));
        TORCH_CHECK((!dequant_scale.has_value()),
                    "when only dtype of x2 is equal to int8, dequantScale should be null", OPS_ERROR(ErrCode::TYPE));
    } else if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        TORCH_CHECK((!antiquant_scale.has_value() && !antiquant_offset.has_value() && !dequant_scale.has_value()),
                    "when neither dtype of x1 or dtype of x2 is equal to int8, "
                    "antiquantScale, antiquantOffset and dequantScale should all be null", OPS_ERROR(ErrCode::TYPE));
    } else {
        TORCH_CHECK(false, "when neither dtype of x1 or dtype of x2 is valid. ", OPS_ERROR(ErrCode::TYPE));
    }

    // check x3 dtype and shape
    if (x3.has_value()) {
        auto output_size = op_infer::array_to_small_vector(x1.sizes());
        output_size[x1.dim() - 1] = x2.size(1);
        // A8W8: dtype of output should be half or bfloat16.
        auto output_dtype = get_output_dtype(x1, dequant_scale);
        const at::Tensor& x3_real = x3.value();
        TORCH_CHECK(x3_real.sizes().equals(output_size), "x3 with shape ", x3_real.sizes(),
                    " doesn't match the output shape ", output_size, OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x3_real.scalar_type() == output_dtype, "x3 with dtype ", x3_real.scalar_type(),
                    " doesn't match the output dtype ", output_dtype, OPS_ERROR(ErrCode::PARAM));
    }

    // check pertoken_scale dtype and shape
    if (pertoken_scale.has_value()) {
        TORCH_CHECK((dequant_scale.has_value()),
                    "when has pertoken_scale, dequantScale shoulden't be null", OPS_ERROR(ErrCode::TYPE));

        const at::Tensor& pertoken_scale_real = pertoken_scale.value();
        TORCH_CHECK(pertoken_scale_real.dim() == 1, "pertoken_scale with shape ", pertoken_scale_real.sizes(),
                    " pertoken_scale dim should be 1.", OPS_ERROR(ErrCode::PARAM));

        auto x1_size = op_infer::array_to_small_vector(x1.sizes());
        int64_t x1_m = 1;
        for (int dim = 0; dim < x1.dim() - 1; dim++) {
            x1_m *= x1_size[dim];
        }
        TORCH_CHECK(x1_m == pertoken_scale_real.size(0), "pertoken_scale with shape ", pertoken_scale_real.sizes(),
                    " doesn't match the input shape ", x1_size, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(pertoken_scale_real.scalar_type() == at::ScalarType::Float,
                    "pertoken_scale with dtype ", pertoken_scale_real.scalar_type(),
                    " doesn't match the output dtype ", at::ScalarType::Float, OPS_ERROR(ErrCode::PARAM));
    }
    // check comm_quant_scale_1, comm_quant_scale_2 dtype and shape
    TORCH_CHECK((comm_quant_scale_1.has_value() && comm_quant_scale_2.has_value()) ||
                (!comm_quant_scale_1.has_value() && !comm_quant_scale_2.has_value()),
                "comm_quant_scale_1 and comm_quant_scale_2 should both be null or not null", OPS_ERROR(ErrCode::TYPE));
    if (comm_quant_scale_1.has_value() && comm_quant_scale_2.has_value()) {
        const at::Tensor& comm_quant_scale_1_real = comm_quant_scale_1.value();
        const at::Tensor& comm_quant_scale_2_real = comm_quant_scale_2.value();
        TORCH_CHECK((comm_quant_scale_1_real.dim() == 2 && comm_quant_scale_2_real.dim() == 2) || (comm_quant_scale_1_real.dim() == 1 &&
                    comm_quant_scale_2_real.dim() == 1), "comm_quant_scale_1 and comm_quant_scale_2 both need to be 1D or 2D, but got: comm_quant_scale_1",
                    comm_quant_scale_1_real.dim(), "D, comm_quant_scale_2", comm_quant_scale_2_real.dim(), "D", OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK((comm_quant_scale_1_real.dim() == 2 && comm_quant_scale_1_real.size(0) == 1 && comm_quant_scale_2_real.size(0) == 1 &&
                    comm_quant_scale_1_real.size(1) == x2.size(1) && comm_quant_scale_2_real.size(1) == x2.size(1)) ||
                    (comm_quant_scale_1_real.dim() == 1 && comm_quant_scale_1_real.size(0) == x2.size(1) && comm_quant_scale_2_real.size(0) == x2.size(1)),
                    "comm_quant_scale_1 and comm_quant_scale_2 shape do not match [1,n] or [n], n=", x2.size(1), ", comm_quant_scale_1 shape: ",
                    comm_quant_scale_1_real.sizes(), ", comm_quant_scale_2 shape: ", comm_quant_scale_2_real.sizes(), OPS_ERROR(ErrCode::PARAM));
        auto output_dtype = get_output_dtype(x1, dequant_scale);
        TORCH_CHECK(comm_quant_scale_1_real.scalar_type() == output_dtype && comm_quant_scale_2_real.scalar_type() == output_dtype,
                    "comm_quant_scale_1 with dtype ", comm_quant_scale_1_real.scalar_type(), "comm_quant_scale_2 with dtype ",
                    comm_quant_scale_2_real.scalar_type(), " doesn't match the output dtype ", output_dtype, OPS_ERROR(ErrCode::PARAM));
    }
}

at::Tensor npu_mm_all_reduce_base(const at::Tensor& x1, const at::Tensor& x2, c10::string_view hcom,
                                  c10::string_view reduce_op, const c10::optional<at::Tensor>& bias,
                                  const c10::optional<at::Tensor>& antiquant_scale,
                                  const c10::optional<at::Tensor>& antiquant_offset,
                                  const c10::optional<at::Tensor>& x3, const c10::optional<at::Tensor>& dequant_scale,
                                  const c10::optional<at::Tensor>& pertoken_scale,
                                  const c10::optional<at::Tensor>& comm_quant_scale_1,
                                  const c10::optional<at::Tensor>& comm_quant_scale_2,
                                  int64_t antiquant_group_size, int64_t comm_turn)
{
    check_params(x1, x2, antiquant_scale, antiquant_offset, x3, dequant_scale, pertoken_scale, comm_quant_scale_1, comm_quant_scale_2);
    // size of last dim of output should be the same as size of last dim of x2
    auto output_size = op_infer::array_to_small_vector(x1.sizes());
    output_size[x1.dim() - 1] = x2.size(1);
    // a8w8: dtype of output should be half.
    auto output_dtype = get_output_dtype(x1, dequant_scale);
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size,
                                                                             x1.options().dtype(output_dtype));
    char* reduce_op_ptr = const_cast<char*>(reduce_op.data());
    char* hcom_ptr = const_cast<char*>(hcom.data());
    const at::Tensor& bias_real = bias.value_or(at::Tensor());
    const at::Tensor& x3_real = x3.value_or(at::Tensor());
    const at::Tensor& comm_quant_scale_1_real = comm_quant_scale_1.value_or(at::Tensor());
    const at::Tensor& comm_quant_scale_2_real = comm_quant_scale_2.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    // a8w8: x1\x2 kChar; a16w8: x2 kChar;
    if (!isIntegralType(x1.scalar_type()) && !isIntegralType(x2.scalar_type())) {
        if (x3.has_value()) {
            EXEC_NPU_CMD(aclnnMatmulAllReduceV2, x1, x2, bias_real, x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, result);
        } else {
            EXEC_NPU_CMD(aclnnMatmulAllReduce, x1, x2, bias_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, result);
        }
    }
    if (isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor& dequant_scale_real = dequant_scale.value_or(at::Tensor());
        if (comm_quant_scale_1.has_value() && comm_quant_scale_2.has_value()) {
            const at::Tensor& pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
            EXEC_NPU_CMD(aclnnQuantMatmulAllReduceV3, x1, x2, bias_real, x3_real, dequant_scale_real,
                         pertoken_scale_real, comm_quant_scale_1_real, comm_quant_scale_2_real, hcom_ptr,
                         reduce_op_ptr, comm_turn, stream_mode, result);
        } else if (pertoken_scale.has_value()) {
            const at::Tensor& pertoken_scale_real = pertoken_scale.value_or(at::Tensor());
            EXEC_NPU_CMD(aclnnQuantMatmulAllReduceV2, x1, x2, bias_real, x3_real, dequant_scale_real, pertoken_scale_real, hcom_ptr, reduce_op_ptr,
                         comm_turn, stream_mode, result);
        } else {
            EXEC_NPU_CMD(aclnnQuantMatmulAllReduce, x1, x2, bias_real, x3_real, dequant_scale_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, result);
        }
    }
    if (!isIntegralType(x1.scalar_type()) && isIntegralType(x2.scalar_type())) {
        const at::Tensor& antiquant_scale_real = antiquant_scale.value_or(at::Tensor());
        const at::Tensor& antiquant_offset_real = antiquant_offset.value_or(at::Tensor());
        EXEC_NPU_CMD(aclnnWeightQuantMatmulAllReduce, x1, x2, bias_real, antiquant_scale_real, antiquant_offset_real,
                     x3_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode, antiquant_group_size, result);
    }

    FLOP_COUNT(FlopCounter::mm_flop, x1, x2);
    return result;
}
}  // namespace op_api
