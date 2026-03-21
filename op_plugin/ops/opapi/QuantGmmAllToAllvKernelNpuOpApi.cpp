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
#include <set>
#include <op_plugin/OpApiInterface.h>
#include <torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h>
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;
static const int QUANT_MODE_PERTENSOR = 1;
static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
static const int MX_QUANT_MODE = 6;
static const int64_t ACL_UNDEFINED = -1;
static const int64_t NON_QUANT = 0;
// world_size
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16, 32, 64, 128, 256};

std::tuple<at::Tensor, at::Tensor> npu_quant_gmm_alltoallv(const at::Tensor &gmm_x,
                                                           const at::Tensor &gmm_weight,
                                                           const at::Tensor &gmm_x_scale,
                                                           const at::Tensor &gmm_weight_scale,
                                                           c10::string_view hcom,
                                                           int64_t ep_world_size,
                                                           at::IntArrayRef send_counts,
                                                           at::IntArrayRef recv_counts,
                                                           int64_t gmm_y_dtype,
                                                           const c10::optional<at::Tensor> &send_counts_tensor,
                                                           const c10::optional<at::Tensor> &recv_counts_tensor,
                                                           const c10::optional<at::Tensor> &mm_x,
                                                           const c10::optional<at::Tensor> &mm_weight,
                                                           const c10::optional<at::Tensor> &mm_x_scale,
                                                           const c10::optional<at::Tensor> &mm_weight_scale,
                                                           const c10::optional<at::Tensor> &comm_quant_scale,
                                                           c10::optional<int64_t> gmm_x_quant_mode,
                                                           c10::optional<int64_t> gmm_weight_quant_mode,
                                                           c10::optional<int64_t> mm_x_quant_mode,
                                                           c10::optional<int64_t> mm_weight_quant_mode,
                                                           c10::optional<int64_t> comm_quant_mode,
                                                           c10::OptionalIntArrayRef group_size,
                                                           c10::optional<int64_t> gmm_x_dtype,
                                                           c10::optional<int64_t> gmm_weight_dtype,
                                                           c10::optional<int64_t> gmm_x_scale_dtype,
                                                           c10::optional<int64_t> gmm_weight_scale_dtype,
                                                           c10::optional<int64_t> mm_x_dtype,
                                                           c10::optional<int64_t> mm_weight_dtype,
                                                           c10::optional<int64_t> mm_x_scale_dtype,
                                                           c10::optional<int64_t> mm_weight_scale_dtype,
                                                           c10::optional<int64_t> comm_quant_dtype,
                                                           c10::optional<int64_t> mm_y_dtype
                                                           )
    {
        // 校验空tensor
        TORCH_CHECK(gmm_x.defined(), "The input tensor gmm_x can not be None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.defined(), "The input tensor gmm_weight can not be None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_x_scale.defined(), "The input tensor gmm_x_scale can not be None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight_scale.defined(), "The input tensor gmm_weight_scale can not be None.", OPS_ERROR(ErrCode::PARAM));
        // dim
        TORCH_CHECK(gmm_x.dim() == DIM_TWO, "The dimension of gmm_x should be 2D, but got ",
            gmm_x.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.dim() == DIM_THREE, "The dimension of gmm_weight should be 3D, but got ",
            gmm_weight.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_x.size(1) == gmm_weight.size(1), "The K-axis in the two inputs of gmm must be equal, but in reality, the K-axis of gmm_x is ",
            gmm_x.size(1), " and the K-axis of gmm_weight is ", gmm_weight.size(1), "." + OPS_ERROR(ErrCode::PARAM));
        // 校验ep_world_size
        TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(ep_world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
            "The world_size should be in [2, 4, 8, 16, 32, 64, 128, 256], but the actual value is ", ep_world_size, OPS_ERROR(ErrCode::VALUE));
        at::Tensor mm_y{nullptr};
        int64_t bsk = 0;
        for (auto &i : recv_counts) {
            bsk += i;
        }
        
        // 处理group_sizes
        at::IntArrayRef group_size_list = group_size.value_or(at::IntArrayRef{});
        int64_t group_sizes = op_plugin::utils::check_and_get_group_size(group_size_list);

        if (mm_x.has_value() && mm_weight.has_value()) {
            TORCH_CHECK(mm_x_scale.has_value() && mm_weight_scale.has_value(),
                "mm_x_scale and mm_weight_scale are required when mm_x and mm_weight are provided",
                OPS_ERROR(ErrCode::PARAM));
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);   // shape (BS， H)
            int64_t n2 = mm_weight_value.size(1);
            // 推导输出mm_y的shape
            aclDataType mm_y_acltype = c10_npu::GetAclDataType(mm_y_dtype.value());
            at::ScalarType mm_y_scalar_dtype = at_npu::native::OpPreparation::convert_to_scalar_type(mm_y_acltype);
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, c10::dtype(mm_y_scalar_dtype));
        }

        int64_t n1 = gmm_weight.size(2);
        // 推导输出gmm_y的shape
        aclDataType gmm_y_acl_type = c10_npu::GetAclDataType(gmm_y_dtype);
        at::ScalarType gmm_y_scalar_type = npu_preparation::convert_to_scalar_type(gmm_y_acl_type);
        auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, c10::dtype(gmm_y_scalar_type));

        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &mm_x_scale_real = mm_x_scale.value_or(at::Tensor());
        const at::Tensor &mm_weight_scale_real = mm_weight_scale.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        char* hcom_ptr = const_cast<char*>(hcom.data());

        int64_t gmm_x_quant_mode_real = gmm_x_quant_mode.has_value() ? gmm_x_quant_mode.value() : QUANT_MODE_PERTENSOR;
        int64_t gmm_weight_quant_mode_real = gmm_weight_quant_mode.has_value() ? gmm_weight_quant_mode.value() : QUANT_MODE_PERTENSOR;
        int64_t mm_x_quant_mode_real = mm_x_quant_mode.has_value() ? mm_x_quant_mode.value() : QUANT_MODE_PERTENSOR;
        int64_t mm_weight_quant_mode_real = mm_weight_quant_mode.has_value() ? mm_weight_quant_mode.value() : QUANT_MODE_PERTENSOR;
        int64_t comm_quant_mode_real = comm_quant_mode.has_value() ? comm_quant_mode.value() : NON_QUANT;
        int64_t comm_quant_dtype_real = comm_quant_dtype.has_value() ? comm_quant_dtype.value() : ACL_UNDEFINED;
        
        // mx量化下scale为fp8_e8m0，需要wrapper包装
        TensorWrapper gmm_x_scale_wrapper = make_wrapper(gmm_x_scale, gmm_x_scale_dtype);
        TensorWrapper gmm_weight_scale_wrapper = make_wrapper(gmm_weight_scale, gmm_weight_scale_dtype);
        TensorWrapper mm_x_scale_wrapper = make_wrapper(mm_x_scale_real, mm_x_scale_dtype);
        TensorWrapper mm_weight_scale_wrapper = make_wrapper(mm_weight_scale_real, mm_weight_scale_dtype);

        TensorWrapper gmm_x_wrapper = make_wrapper(gmm_x, gmm_x_dtype);
        TensorWrapper gmm_weight_wrapper = make_wrapper(gmm_weight, gmm_weight_dtype);
        TensorWrapper mm_x_wrapper = make_wrapper(mm_x_real, mm_x_dtype);
        TensorWrapper mm_weight_wrapper = make_wrapper(mm_weight_real, mm_weight_dtype);

        bool trans_gmm_weight = false;
        bool trans_mm_weight = false;

        EXEC_NPU_CMD(aclnnQuantGroupedMatMulAlltoAllv,
                     gmm_x_wrapper,
                     gmm_weight_wrapper,
                     gmm_x_scale_wrapper,
                     gmm_weight_scale_wrapper,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_wrapper,
                     mm_weight_wrapper,
                     mm_x_scale_wrapper,
                     mm_weight_scale_wrapper,
                     comm_quant_scale,
                     gmm_x_quant_mode_real,
                     gmm_weight_quant_mode_real,
                     mm_x_quant_mode_real,
                     mm_weight_quant_mode_real,
                     comm_quant_mode_real,
                     comm_quant_dtype_real,
                     group_sizes,
                     hcom_ptr,
                     ep_world_size,
                     send_counts,
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     gmm_y,
                     mm_y);
        return std::tie(gmm_y, mm_y);
    }
} // namespace op_api