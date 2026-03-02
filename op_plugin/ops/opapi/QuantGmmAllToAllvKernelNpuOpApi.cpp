// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

using npu_preparation = at_npu::native::OpPreparation;
static const int QUANT_MODE_PERTENSOR = 1;
static const int DIM_TWO = 2;
static const int DIM_THREE = 3;

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
                                                           const c10::optional<at::Tensor> &gmm_x_offset,
                                                           const c10::optional<at::Tensor> &gmm_weight_offset,
                                                           const c10::optional<at::Tensor> &mm_x_offset,
                                                           const c10::optional<at::Tensor> &mm_weight_offset,
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
        // quant mode
        TORCH_CHECK(gmm_x_quant_mode.value_or(QUANT_MODE_PERTENSOR) == QUANT_MODE_PERTENSOR, "gmm_x_quant_mode only support 1 (QUANT_MODE_PERTENSOR), but got ",
            gmm_x_quant_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(gmm_weight_quant_mode.value_or(QUANT_MODE_PERTENSOR) == QUANT_MODE_PERTENSOR, "gmm_weight_quant_mode only support 1 (QUANT_MODE_PERTENSOR), but got ",
            gmm_weight_quant_mode, OPS_ERROR(ErrCode::PARAM));

        TORCH_CHECK(comm_quant_mode.value_or(0) == 0, "comm_quant_mode only support 0, but got ",
            comm_quant_mode, OPS_ERROR(ErrCode::PARAM));
        // dim
        TORCH_CHECK(gmm_x.dim() == DIM_TWO, "The dimension of gmm_x should be 2D, but got ",
            gmm_x.dim(), OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.dim() == DIM_THREE, "The dimension of gmm_weight should be 3D, but got ",
            gmm_weight.dim(), OPS_ERROR(ErrCode::PARAM));

        bool trans_gmm_weight = false;
        bool trans_mm_weight = false;

        at::Tensor mm_y{nullptr};
        int64_t bsk = 0;
        for (auto &i : recv_counts) {
            bsk += i;
        }
        int64_t n1 = gmm_weight.size(2);
        if (mm_x.has_value() && mm_weight.has_value()) {
            TORCH_CHECK(mm_x_quant_mode == QUANT_MODE_PERTENSOR, "mm_x_quant_mode only support 1 (QUANT_MODE_PERTENSOR), but got ",
                mm_x_quant_mode, OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_weight_quant_mode == QUANT_MODE_PERTENSOR, "mm_weight_quant_mode only support 1 (QUANT_MODE_PERTENSOR), but got ",
                mm_weight_quant_mode, OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_x_scale.has_value() && mm_weight_scale.has_value(),
                "mm_x_scale and mm_weight_scale are required when mm_x and mm_weight are provided",
                OPS_ERROR(ErrCode::PARAM));
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);   // shape (BS， H)
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1);
            at::ScalarType mm_y_scalar_type = mm_y_dtype.has_value()
                ? npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(mm_y_dtype.value()))
                : mm_x_value.scalar_type();
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format(
                {bs, n2}, mm_x_value.options().dtype(mm_y_scalar_type));
        }

        aclDataType gmm_y_acl_type = c10_npu::GetAclDataType(gmm_y_dtype);
        at::ScalarType gmm_y_scalar_type = npu_preparation::convert_to_scalar_type(gmm_y_acl_type);
        auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format(
            {bsk, n1}, gmm_x.options().dtype(gmm_y_scalar_type));
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        char* hcom_ptr = const_cast<char*>(hcom.data());
        
        TensorWrapper gmm_x_wrapper = {gmm_x, gmm_x_dtype.has_value()
            ? c10_npu::GetAclDataType(gmm_x_dtype.value())
            : npu_preparation::convert_to_acl_data_type(gmm_x.scalar_type())};
        TensorWrapper gmm_weight_wrapper = {gmm_weight, gmm_weight_dtype.has_value()
            ? c10_npu::GetAclDataType(gmm_weight_dtype.value())
            : npu_preparation::convert_to_acl_data_type(gmm_weight.scalar_type())};
        TensorWrapper mm_x_wrapper = {mm_x.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_x_dtype.value_or(0))};
        TensorWrapper mm_weight_wrapper = {mm_weight.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_weight_dtype.value_or(0))};
        TensorWrapper gmm_x_scale_wrapper = {gmm_x_scale, gmm_x_scale_dtype.has_value()
            ? c10_npu::GetAclDataType(gmm_x_scale_dtype.value())
            : npu_preparation::convert_to_acl_data_type(gmm_x_scale.scalar_type())};
        TensorWrapper gmm_weight_scale_wrapper = {gmm_weight_scale, gmm_weight_scale_dtype.has_value()
            ? c10_npu::GetAclDataType(gmm_weight_scale_dtype.value())
            : npu_preparation::convert_to_acl_data_type(gmm_weight_scale.scalar_type())};
        TensorWrapper mm_x_scale_wrapper = {mm_x_scale.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_x_scale_dtype.value_or(0))};
        TensorWrapper mm_weight_scale_wrapper = {mm_weight_scale.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_weight_scale_dtype.value_or(0))};
        TensorWrapper gmm_x_offset_wrapper = {gmm_x_offset.value_or(at::Tensor()), c10_npu::GetAclDataType(gmm_x_dtype.value_or(0))};
        TensorWrapper gmm_weight_offset_wrapper = {gmm_weight_offset.value_or(at::Tensor()), c10_npu::GetAclDataType(gmm_weight_dtype.value_or(0))};
        TensorWrapper mm_x_offset_wrapper = {mm_x_offset.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_x_dtype.value_or(0))};
        TensorWrapper mm_weight_offset_wrapper = {mm_weight_offset.value_or(at::Tensor()), c10_npu::GetAclDataType(mm_weight_dtype.value_or(0))};
        TensorWrapper comm_quant_scale_wrapper = {comm_quant_scale.value_or(at::Tensor()), c10_npu::GetAclDataType(comm_quant_dtype.value_or(0))};
        TensorWrapper gmm_y_wrapper = {gmm_y, c10_npu::GetAclDataType(gmm_y_dtype)};
        TensorWrapper mm_y_wrapper = {mm_y, c10_npu::GetAclDataType(mm_y_dtype.value_or(0))};

        int64_t gmm_x_quant_mode_real = gmm_x_quant_mode.value_or(1);
        int64_t gmm_weight_quant_mode_real = gmm_weight_quant_mode.value_or(1);
        int64_t mm_x_quant_mode_real = mm_x_quant_mode.value_or(1);
        int64_t mm_weight_quant_mode_real = mm_weight_quant_mode.value_or(1);
        int64_t comm_quant_mode_real = comm_quant_mode.value_or(0);
        int64_t comm_quant_dtype_real = comm_quant_dtype.value_or(0);
        // 当前暂不支持pergroup量化，groupsize直接赋值为0；
        int64_t group_size_real = 0;

        EXEC_NPU_CMD(aclnnQuantGroupedMatMulAlltoAllv,
                     gmm_x_wrapper,
                     gmm_weight_wrapper,
                     gmm_x_scale_wrapper,
                     gmm_weight_scale_wrapper,
                     gmm_x_offset_wrapper,
                     gmm_weight_offset_wrapper,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_wrapper,
                     mm_weight_wrapper,
                     mm_x_scale_wrapper,
                     mm_weight_scale_wrapper,
                     mm_x_offset_wrapper,
                     mm_weight_offset_wrapper,
                     comm_quant_scale_wrapper,
                     gmm_x_quant_mode_real,
                     gmm_weight_quant_mode_real,
                     mm_x_quant_mode_real,
                     mm_weight_quant_mode_real,
                     comm_quant_mode_real,
                     comm_quant_dtype_real,
                     group_size_real,
                     hcom_ptr,
                     ep_world_size,
                     send_counts,
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     gmm_y_wrapper,
                     mm_y_wrapper);
        return std::tie(gmm_y, mm_y);
    }
} // namespace op_api