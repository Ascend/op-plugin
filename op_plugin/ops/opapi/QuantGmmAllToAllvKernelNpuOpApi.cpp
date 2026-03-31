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
// dim
static const int DIM_ONE = 1;
static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
static const int DIM_FOUR = 4;
// quant mode
static const int MX_QUANT_MODE = 6;
static const int NO_QUANT_MODE = 0;
static const int PERTENSOR_QUANT_MODE = 1;
static const int64_t ACL_UNDEFINED = -1;
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
        // gmm_x_quant_mode
        TORCH_CHECK(gmm_x_quant_mode.has_value(),
            "The input gmm_x_quant_mode must be provided, but got None.",
            OPS_ERROR(ErrCode::PARAM));
        int64_t gmm_x_quant_mode_real = gmm_x_quant_mode.value();
        // gmm_weight_quant_mode
        TORCH_CHECK(gmm_weight_quant_mode.has_value(),
            "The input gmm_weight_quant_mode must be provided, but got None.",
            OPS_ERROR(ErrCode::PARAM));
        int64_t gmm_weight_quant_mode_real = gmm_weight_quant_mode.value();
        // gmm_x
        TORCH_CHECK(gmm_x.defined(), "The input tensor gmm_x can not be None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_x.dim() == DIM_TWO, "The dim of gmm_x should be 2D, but got ",
            gmm_x.dim(), OPS_ERROR(ErrCode::PARAM));
        if (gmm_x_quant_mode_real == PERTENSOR_QUANT_MODE) {
            TORCH_CHECK(gmm_x_dtype.has_value(),
                "The input gmm_x_dtype must be provided for pertensor quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(gmm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                "The input gmm_x_dtype should be hifloat8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(gmm_x_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(gmm_x.scalar_type() == at::ScalarType::Byte || gmm_x.scalar_type() == at::ScalarType::Char,
                "The input gmm_x tensor dtype should be uint8 or int8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(static_cast<int64_t>(gmm_x.scalar_type())),
                ".",
                OPS_ERROR(ErrCode::TYPE));
        } else if (gmm_x_quant_mode_real == MX_QUANT_MODE) {
            if (gmm_x_dtype.has_value()) {
                bool is_supported = (gmm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                                     gmm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2) ||
                                     gmm_x_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn) ||
                                     gmm_x_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e5m2));
                TORCH_CHECK(is_supported,
                    "The input gmm_x_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                    op_plugin::utils::DTypeToString(gmm_x_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
            }
        }
        // gmm_weight
        TORCH_CHECK(gmm_weight.defined(), "The input tensor gmm_weight can not be None.", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight.dim() == DIM_THREE,
            "The dim of gmm_weight should be 3, but got ",
            gmm_weight.dim(),
            ".",
            OPS_ERROR(ErrCode::PARAM));
        if (gmm_weight_quant_mode_real == PERTENSOR_QUANT_MODE) {
            TORCH_CHECK(gmm_weight_dtype.has_value(),
                "The input gmm_weight_dtype must be provided for pertensor quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(gmm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                "The input gmm_weight_dtype should be hifloat8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(gmm_weight_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(
                gmm_weight.scalar_type() == at::ScalarType::Byte || gmm_weight.scalar_type() == at::ScalarType::Char,
                "The input gmm_weight tensor dtype should be uint8 or int8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(static_cast<int64_t>(gmm_weight.scalar_type())),
                ".",
                OPS_ERROR(ErrCode::TYPE));
        } else if (gmm_weight_quant_mode_real == MX_QUANT_MODE) {
            if (gmm_weight_dtype.has_value()) {
                bool is_supported = (gmm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                                     gmm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2) ||
                                     gmm_weight_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn) ||
                                     gmm_weight_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e5m2));
                TORCH_CHECK(is_supported,
                    "The input gmm_weight_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                    op_plugin::utils::DTypeToString(gmm_weight_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
            }
        }
        // gmm_x_scale
        TORCH_CHECK(gmm_x_scale.defined(), "The input tensor gmm_x_scale can not be None.", OPS_ERROR(ErrCode::PARAM));
        int gmm_x_scale_dim = (gmm_x_quant_mode_real == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE;
        TORCH_CHECK(gmm_x_scale.dim() == gmm_x_scale_dim,
            "The dim of gmm_x_scale should be ",
            gmm_x_scale_dim,
            ", but got ",
            gmm_x_scale.dim(),
            ".",
            OPS_ERROR(ErrCode::PARAM));
        if (gmm_x_quant_mode_real == PERTENSOR_QUANT_MODE) {
            if (gmm_x_scale_dtype.has_value()) {
                TORCH_CHECK(gmm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT) ||
                                gmm_x_scale_dtype.value() == static_cast<int64_t>(at::ScalarType::Float),
                    "The input gmm_x_scale_dtype should be float32 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(gmm_x_scale_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
            }
            TORCH_CHECK(gmm_x_scale.scalar_type() == at::ScalarType::Float,
                "The input gmm_x_scale tensor dtype should be float32 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(static_cast<int64_t>(gmm_x_scale.scalar_type())),
                ".",
                OPS_ERROR(ErrCode::TYPE));
        } else if (gmm_x_quant_mode_real == MX_QUANT_MODE) {
            TORCH_CHECK(gmm_x_scale_dtype.has_value(),
                "The input gmm_x_scale_dtype must be provided for mx quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(gmm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                "The input gmm_x_scale_dtype should be float8_e8m0 for mx quant mode, but got ",
                op_plugin::utils::DTypeToString(gmm_x_scale_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
        }
        // gmm_weight_scale
        TORCH_CHECK(gmm_weight_scale.defined(), "The input tensor gmm_weight_scale can not be None.", OPS_ERROR(ErrCode::PARAM));
        int gmm_weight_scale_dim = (gmm_weight_quant_mode_real == MX_QUANT_MODE) ? DIM_FOUR : DIM_ONE;
        TORCH_CHECK(gmm_weight_scale.dim() == gmm_weight_scale_dim,
            "The dim of gmm_weight_scale should be ",
            gmm_weight_scale_dim,
            ", but got ",
            gmm_weight_scale.dim(),
            ".",
            OPS_ERROR(ErrCode::PARAM));
        if (gmm_weight_quant_mode_real == PERTENSOR_QUANT_MODE) {
            if (gmm_weight_scale_dtype.has_value()) {
                TORCH_CHECK(gmm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT) ||
                                gmm_weight_scale_dtype.value() == static_cast<int64_t>(at::ScalarType::Float),
                    "The input gmm_weight_scale_dtype should be float32 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(gmm_weight_scale_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
            }
            TORCH_CHECK(gmm_weight_scale.scalar_type() == at::ScalarType::Float,
                "The input gmm_weight_scale tensor dtype should be float32 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(static_cast<int64_t>(gmm_weight_scale.scalar_type())),
                ".",
                OPS_ERROR(ErrCode::TYPE));
        } else if (gmm_weight_quant_mode_real == MX_QUANT_MODE) {
            TORCH_CHECK(gmm_weight_scale_dtype.has_value(),
                "The input gmm_weight_scale_dtype must be provided for mx quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(gmm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                "The input gmm_weight_scale_dtype should be float8_e8m0 for mx quant mode, but got ",
                op_plugin::utils::DTypeToString(gmm_weight_scale_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
        }

        // ep_world_size
        TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(ep_world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
            "The world_size should be in [2, 4, 8, 16, 32, 64, 128, 256], but the actual value is ",
            ep_world_size,
            OPS_ERROR(ErrCode::VALUE));

        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &mm_x_scale_real = mm_x_scale.value_or(at::Tensor());
        const at::Tensor &mm_weight_scale_real = mm_weight_scale.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        char* hcom_ptr = const_cast<char*>(hcom.data());

        at::Tensor mm_y{nullptr};
        int64_t mm_x_quant_mode_real = NO_QUANT_MODE;
        int64_t mm_weight_quant_mode_real = NO_QUANT_MODE;
        int64_t bsk = 0;
        for (auto &i : recv_counts) {
            bsk += i;
        }
        
        // 处理group_sizes
        at::IntArrayRef group_size_list = group_size.value_or(at::IntArrayRef{});
        int64_t group_sizes = op_plugin::utils::check_and_get_group_size(group_size_list);

        // mm_x and mm_weight consistency check
        bool mm_x_has_value = mm_x.has_value();
        bool mm_weight_has_value = mm_weight.has_value();
        if (mm_x_has_value != mm_weight_has_value) {
            std::string error_msg = "The input mm_x and mm_weight must be both provided or both None, but the input ";
            if (mm_x_has_value && !mm_weight_has_value) {
                error_msg += "mm_x is provided and mm_weight is None.";
            } else if (!mm_x_has_value && mm_weight_has_value) {
                error_msg += "mm_weight is provided and mm_x is None.";
            }
            TORCH_CHECK(false, error_msg, OPS_ERROR(ErrCode::PARAM));
        }
        // mmx and mmweight is not exist
        if (!mm_x.has_value() && !mm_weight.has_value()) {
            TORCH_CHECK(!mm_x_quant_mode.has_value(),
                "The input mm_x_quant_mode should be None when mm_x is not provided.",
                OPS_ERROR(ErrCode::PARAM));

            TORCH_CHECK(!mm_weight_quant_mode.has_value(),
                "The input mm_weight_quant_mode should be None when mm_x is not provided.",
                OPS_ERROR(ErrCode::PARAM));
            
        }
        // mmx and mmweight is exist
        if (mm_x.has_value() && mm_weight.has_value()) {
            TORCH_CHECK(mm_y_dtype.has_value(),
                "The input mm_y_dtype must be provided when mm_x and mm_weight are present.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_x_quant_mode.has_value(),
                "The input mm_x_quant_mode must be provided when mm_x is present.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_weight_quant_mode.has_value(),
                "The input mm_weight_quant_mode must be provided when mm_weight is present.",
                OPS_ERROR(ErrCode::PARAM));

            mm_x_quant_mode_real = mm_x_quant_mode.value();
            mm_weight_quant_mode_real = mm_weight_quant_mode.value();
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            // mm_x
            TORCH_CHECK(mm_x_value.dim() == DIM_TWO,
                "The dim of mm_x should be 2, but got ",
                mm_x_value.dim(),
                ".",
                OPS_ERROR(ErrCode::PARAM));
            if (mm_x_quant_mode_real == PERTENSOR_QUANT_MODE) {
                TORCH_CHECK(mm_x_dtype.has_value(),
                    "The input mm_x_dtype must be provided for pertensor quant mode.",
                    OPS_ERROR(ErrCode::PARAM));
                TORCH_CHECK(mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                    "The input mm_x_dtype should be hifloat8 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(mm_x_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
                TORCH_CHECK(mm_x_value.scalar_type() == at::ScalarType::Byte ||
                                mm_x_value.scalar_type() == at::ScalarType::Char,
                    "The input mm_x tensor dtype should be uint8 or int8 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(static_cast<int64_t>(mm_x_value.scalar_type())),
                    ".",
                    OPS_ERROR(ErrCode::TYPE));
            } else if (mm_x_quant_mode_real == MX_QUANT_MODE) {
                if (mm_x_dtype.has_value()) {
                    TORCH_CHECK(mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                                    mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2) ||
                                    mm_x_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn) ||
                                    mm_x_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e5m2),
                        "The input mm_x_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                        op_plugin::utils::DTypeToString(mm_x_dtype.value()),
                        ".",
                        OPS_ERROR(ErrCode::VALUE));
                }
            }
            // mm_weight
            TORCH_CHECK(mm_weight_value.dim() == DIM_TWO,
                "The dim of mm_weight should be 2, but got ",
                mm_weight_value.dim(),
                ".",
                OPS_ERROR(ErrCode::PARAM));
            if (mm_weight_quant_mode_real == PERTENSOR_QUANT_MODE) {
                TORCH_CHECK(mm_weight_dtype.has_value(),
                    "The input mm_weight_dtype must be provided for pertensor quant mode.",
                    OPS_ERROR(ErrCode::PARAM));
                TORCH_CHECK(mm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                    "The input mm_weight_dtype should be hifloat8 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(mm_weight_dtype.value()),
                    ".",
                    OPS_ERROR(ErrCode::VALUE));
                TORCH_CHECK(mm_weight_value.scalar_type() == at::ScalarType::Byte ||
                                mm_weight_value.scalar_type() == at::ScalarType::Char,
                    "The input mm_weight tensor dtype should be uint8 or int8 for pertensor quant mode, but got ",
                    op_plugin::utils::DTypeToString(static_cast<int64_t>(mm_weight_value.scalar_type())),
                    ".",
                    OPS_ERROR(ErrCode::TYPE));
            } else if (mm_weight_quant_mode_real == MX_QUANT_MODE) {
                if (mm_weight_dtype.has_value()) {
                    TORCH_CHECK(mm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                                    mm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2) ||
                                    mm_weight_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e4m3fn) ||
                                    mm_weight_dtype.value() == static_cast<int64_t>(at::ScalarType::Float8_e5m2),
                        "The input mm_weight_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                        op_plugin::utils::DTypeToString(mm_weight_dtype.value()),
                        ".",
                        OPS_ERROR(ErrCode::VALUE));
                }
            }
            int64_t bs = mm_x_value.size(0);   // shape (BS， H)
            int64_t n2 = mm_weight_value.size(1);
            // mm_x_scale
            if (mm_x_scale.has_value()) {
                const at::Tensor &mm_x_scale_val = mm_x_scale.value();
                TORCH_CHECK(mm_x_scale_val.dim() == ((mm_x_quant_mode_real == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                    "The dim of mm_x_scale should be ",
                    ((mm_x_quant_mode_real == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                    " for quant_mode ",
                    mm_x_quant_mode_real,
                    ", but got ",
                    mm_x_scale_val.dim(),
                    ".",
                    OPS_ERROR(ErrCode::PARAM));

                if (mm_x_quant_mode_real == PERTENSOR_QUANT_MODE) {
                    if (mm_x_scale_dtype.has_value()) {
                        TORCH_CHECK(mm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT) ||
                                        mm_x_scale_dtype.value() == static_cast<int64_t>(at::ScalarType::Float),
                            "The input mm_x_scale_dtype should be float32 for pertensor quant mode, but got ",
                            op_plugin::utils::DTypeToString(mm_x_scale_dtype.value()),
                            ".",
                            OPS_ERROR(ErrCode::VALUE));
                    }
                    TORCH_CHECK(mm_x_scale_val.scalar_type() == at::ScalarType::Float,
                        "The input mm_x_scale tensor dtype should be float32 for pertensor quant mode, but got ",
                        op_plugin::utils::DTypeToString(static_cast<int64_t>(mm_x_scale_val.scalar_type())),
                        ".",
                        OPS_ERROR(ErrCode::TYPE));
                } else {
                    TORCH_CHECK(mm_x_scale_dtype.has_value(),
                        "mm_x_scale_dtype must be provided for mx quant mode.",
                        OPS_ERROR(ErrCode::PARAM));
                    TORCH_CHECK(mm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                        "mm_x_scale_dtype should be float8_e8m0 for mx quant mode, but got ",
                        op_plugin::utils::DTypeToString(mm_x_scale_dtype.value()),
                        ".",
                        OPS_ERROR(ErrCode::VALUE));
                }
            }
            if (mm_weight_scale.has_value()) {
                const at::Tensor &mm_weight_scale_val = mm_weight_scale.value();
                TORCH_CHECK(
                    mm_weight_scale_val.dim() == ((mm_weight_quant_mode_real == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                    "The dim of mm_weight_scale should be ",
                    ((mm_weight_quant_mode_real == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                    " for quant_mode ",
                    mm_weight_quant_mode_real,
                    ", but got ",
                    mm_weight_scale_val.dim(),
                    ".",
                    OPS_ERROR(ErrCode::PARAM));

                if (mm_weight_quant_mode_real == PERTENSOR_QUANT_MODE) {
                    if (mm_weight_scale_dtype.has_value()) {
                        TORCH_CHECK(mm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT) ||
                                        mm_weight_scale_dtype.value() == static_cast<int64_t>(at::ScalarType::Float),
                            "The input mm_weight_scale_dtype should be float32 for pertensor quant mode, but got ",
                            op_plugin::utils::DTypeToString(mm_weight_scale_dtype.value()),
                            ".",
                            OPS_ERROR(ErrCode::VALUE));
                    }
                    TORCH_CHECK(mm_weight_scale_val.scalar_type() == at::ScalarType::Float,
                        "The input mm_weight_scale tensor dtype should be float32 for pertensor quant mode, but got ",
                        op_plugin::utils::DTypeToString(static_cast<int64_t>(mm_weight_scale_val.scalar_type())),
                        ".",
                        OPS_ERROR(ErrCode::TYPE));
                } else if (mm_weight_quant_mode_real == MX_QUANT_MODE) {
                    TORCH_CHECK(mm_weight_scale_dtype.has_value(),
                        "The input mm_weight_scale_dtype must be provided for mx quant mode.",
                        OPS_ERROR(ErrCode::PARAM));
                    TORCH_CHECK(mm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
                        "The input mm_weight_scale_dtype should be float8_e8m0 for mx quant mode, but got ",
                        op_plugin::utils::DTypeToString(mm_weight_scale_dtype.value()),
                        ".",
                        OPS_ERROR(ErrCode::VALUE));
                }
            }
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
        
        int64_t comm_quant_mode_real = comm_quant_mode.has_value() ? comm_quant_mode.value() : NO_QUANT_MODE;

        if (comm_quant_dtype.has_value()) {
            TORCH_CHECK(comm_quant_dtype.value() == static_cast<int64_t>(c10_npu::DType::INT64) ||
                            comm_quant_dtype.value() == static_cast<int64_t>(at::ScalarType::Long),
                "The input comm_quant_dtype should be int64 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(comm_quant_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
        }
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