// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
    using namespace op_infer;
    namespace {
    constexpr int64_t BLOCK_SIZE_BASE_NUM = 32;
    constexpr int64_t ALIGN_NUM = 2;
    constexpr int64_t FP4_IN_UINT8_NUM = 2;
    constexpr int64_t MIN_INPUT_DIM = 1;
    constexpr int64_t MAX_INPUT_DIM = 7;
    } // namespace

    tensor_list npu_add_rms_norm_dynamic_mx_quant(const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &gamma,
                                                  const c10::optional<at::Tensor> &beta, double epsilon, int64_t scale_alg,
                                                  c10::string_view round_mode, int64_t dst_type)
    {
        // 输出Tensor准备
        at::Tensor y;
        at::Tensor x_out;
        at::Tensor mxscale;
        at::Tensor rstd;

        // 参数检查
        TORCH_CHECK(x1.dim() >= MIN_INPUT_DIM && x1.dim() <= MAX_INPUT_DIM, "The x1 should be in 1~7D" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x2.dim() >= MIN_INPUT_DIM && x2.dim() <= MAX_INPUT_DIM, "The x2 should be in 1~7D" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x1.sizes() == x2.sizes(), "The shape of x1 and x2 must be the same" + OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(x1.requires_grad() == x2.requires_grad(),
                    "The requires_grad of x1 and x2 must be consistent" + OPS_ERROR(ErrCode::PARAM));

        static const bool is_available = check_aclnn_kernel_available("aclnnAddRmsNormDynamicMxQuant");
        TORCH_CHECK(is_available,
                    "Current CANN version do not support this api. Please try to update the version of CANN."
                    + OPS_ERROR(ErrCode::PARAM));
        
        // 类型推断
        auto y_shape = array_to_small_vector(x1.sizes());
        auto mxscale_shape = array_to_small_vector(x1.sizes());
        mxscale_shape.emplace_back(ALIGN_NUM);

        // y shape&dtype 推导
        aclDataType y_acltype;
        bool special_output_type = (dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E2M1) ||
                                dst_type == static_cast<int64_t>(c10_npu::DType::FLOAT4_E1M2));
        ASCEND_LOGI("[npu_add_rms_norm_dynamic_mx_quant]: Getting aclTensor y dtype by Parameter(dst_type): %ld", dst_type);
        if (special_output_type) {
            int64_t y_last_dim_val = y_shape[x1.dim() - 1];
            TORCH_CHECK(y_last_dim_val % FP4_IN_UINT8_NUM == 0,
                        "The last dim input shape must be divisible by 2 if "
                        "y dtype is torch_npu.float4_e2m1fn_x2 or torch_npu.float4_e1m2" + OPS_ERROR(ErrCode::PARAM));
            // Pytorch2.8之前最小单位是8位，不支持FP4类型，就将两个FP4合成一个FP8
            y_shape[x1.dim() - 1] = y_last_dim_val / FP4_IN_UINT8_NUM;
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::ScalarType::Byte);
            y_acltype = c10_npu::GetAclDataType(dst_type);
        } else {
            y_acltype = c10_npu::GetAclDataType(dst_type);
            at::ScalarType scalar_dtype = npu_preparation::convert_to_scalar_type(y_acltype);
            y = npu_preparation::apply_tensor_without_format(y_shape, c10::dtype(scalar_dtype));
        }

        // x_out shape 推导
        auto x_out_shape = x1.sizes();
        auto x_out_dtype = x1.scalar_type();
        x_out = npu_preparation::apply_tensor_without_format(x_out_shape, x1.options().dtype(x_out_dtype));

        // mxscale shape 推导
        int64_t last_axis_change = x1.dim() - 1;
        int64_t last_dim_size = CeilDiv(mxscale_shape[last_axis_change], BLOCK_SIZE_BASE_NUM);
        last_dim_size = (last_dim_size + ALIGN_NUM - 1) / ALIGN_NUM;
        mxscale_shape[last_axis_change] = last_dim_size;
        mxscale = npu_preparation::apply_tensor_without_format(mxscale_shape, c10::dtype(at::ScalarType::Byte));

        // rstd shape 推导
        bool output_rstd = x1.requires_grad() && x2.requires_grad();
        if (output_rstd) {
            auto output_size = rms_norm_npu_output_size(x1, gamma);
            rstd = npu_preparation::apply_tensor_without_format(output_size[1], x1.options().dtype(at::kFloat));
        } else {
            rstd = at::empty({0}, x1.options().dtype(at::kFloat));
        }

        // 调用NPU原生算子执行
        char *round_mode_ptr = const_cast<char *>(round_mode.data());
        TensorWrapper y_wrapper = {y, y_acltype};
        TensorWrapper mxscale_wrapper = {mxscale, aclDataType::ACL_FLOAT8_E8M0};

        EXEC_NPU_CMD(aclnnAddRmsNormDynamicMxQuant, x1, x2, gamma, beta, epsilon, scale_alg,
                     round_mode_ptr, y_acltype, output_rstd, y_wrapper, x_out, mxscale_wrapper, rstd);
        
        return std::make_tuple(y, x_out, mxscale, rstd);
    }
} // namespace op_api