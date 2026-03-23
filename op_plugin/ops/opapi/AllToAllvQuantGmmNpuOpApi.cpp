// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#include <set>
#include "op_plugin/OpApiInterface.h"
#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/OpUtils.h"

namespace op_api {
// dim
static const int DIM_ONE = 1;
static const int DIM_TWO = 2;
static const int DIM_THREE = 3;
// quant mode
static const int NO_QUANT_MODE = 0;
static const int PERTENSOR_QUANT_MODE = 1;
static const int MX_QUANT_MODE = 6;
// support list
const std::set<int> SUPPORT_WORLD_SIZE_LIST{2, 4, 8, 16, 32, 64, 128, 256};

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_alltoallv_quant_gmm(const at::Tensor &gmm_x,
    const at::Tensor &gmm_weight, const at::Tensor &gmm_x_scale, const at::Tensor &gmm_weight_scale,
    c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts,
    int64_t gmm_y_dtype, const c10::optional<at::Tensor> &send_counts_tensor,
    const c10::optional<at::Tensor> &recv_counts_tensor, const c10::optional<at::Tensor> &mm_x,
    const c10::optional<at::Tensor> &mm_weight, const c10::optional<at::Tensor> &mm_x_scale,
    const c10::optional<at::Tensor> &mm_weight_scale, c10::optional<int64_t> gmm_x_quant_mode,
    c10::optional<int64_t> gmm_weight_quant_mode, c10::optional<int64_t> mm_x_quant_mode,
    c10::optional<int64_t> mm_weight_quant_mode, bool permute_out_flag, c10::OptionalIntArrayRef group_size,
    c10::optional<int64_t> gmm_x_dtype, c10::optional<int64_t> gmm_weight_dtype,
    c10::optional<int64_t> gmm_x_scale_dtype, c10::optional<int64_t> gmm_weight_scale_dtype,
    c10::optional<int64_t> mm_x_dtype, c10::optional<int64_t> mm_weight_dtype, c10::optional<int64_t> mm_x_scale_dtype,
    c10::optional<int64_t> mm_weight_scale_dtype, c10::optional<int64_t> mm_y_dtype)
{
    // gmm_x_quant_mode
    TORCH_CHECK(gmm_x_quant_mode.has_value(),
        "The input gmm_x_quant_mode must be provided, but got None.",
        OPS_ERROR(ErrCode::PARAM));
    int64_t gmm_x_quant_mode_value = gmm_x_quant_mode.value();
    // gmm_weight_quant_mode
    TORCH_CHECK(gmm_weight_quant_mode.has_value(),
        "The input gmm_weight_quant_mode must be provided, but got None.",
        OPS_ERROR(ErrCode::PARAM));
    int64_t gmm_weight_quant_mode_value = gmm_weight_quant_mode.value();
    // gmm_x
    TORCH_CHECK(gmm_x.defined(), "The input tensor gmm_x cannot be None.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(
        gmm_x.dim() == DIM_TWO, "The dim of gmm_x should be 2, but got ", gmm_x.dim(), ".", OPS_ERROR(ErrCode::PARAM));
    if (gmm_x_quant_mode_value == PERTENSOR_QUANT_MODE) {
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
    } else if (gmm_x_quant_mode_value == MX_QUANT_MODE) {
        TORCH_CHECK(gmm_x_dtype.has_value(),
            "The input gmm_x_dtype must be provided for mx quant mode.",
            OPS_ERROR(ErrCode::PARAM));
        bool is_supported = (gmm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                             gmm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2));
        TORCH_CHECK(is_supported,
            "The input gmm_x_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
            op_plugin::utils::DTypeToString(gmm_x_dtype.value()),
            ".",
            OPS_ERROR(ErrCode::VALUE));
    }
    // gmm_weight
    TORCH_CHECK(gmm_weight.defined(), "The input tensor gmm_weight cannot be None.", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(gmm_weight.dim() == DIM_THREE,
        "The dim of gmm_weight should be 3, but got ",
        gmm_weight.dim(),
        ".",
        OPS_ERROR(ErrCode::PARAM));
    if (gmm_weight_quant_mode_value == PERTENSOR_QUANT_MODE) {
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
    } else if (gmm_weight_quant_mode_value == MX_QUANT_MODE) {
        TORCH_CHECK(gmm_weight_dtype.has_value(),
            "The input gmm_weight_dtype must be provided for mx quant mode.",
            OPS_ERROR(ErrCode::PARAM));
        bool is_supported = (gmm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                             gmm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2));
        TORCH_CHECK(is_supported,
            "The input gmm_weight_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
            op_plugin::utils::DTypeToString(gmm_weight_dtype.value()),
            ".",
            OPS_ERROR(ErrCode::VALUE));
    }
    // gmm_x_scale
    TORCH_CHECK(gmm_x_scale.defined(), "The input tensor gmm_x_scale cannot be None.", OPS_ERROR(ErrCode::PARAM));
    int gmm_x_scale_dim = (gmm_x_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE;
    TORCH_CHECK(gmm_x_scale.dim() == gmm_x_scale_dim,
        "The dim of gmm_x_scale should be ",
        gmm_x_scale_dim,
        ", but got ",
        gmm_x_scale.dim(),
        ".",
        OPS_ERROR(ErrCode::PARAM));
    if (gmm_x_quant_mode_value == PERTENSOR_QUANT_MODE) {
        if (gmm_x_scale_dtype.has_value()) {
            TORCH_CHECK(gmm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT),
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
    } else if (gmm_x_quant_mode_value == MX_QUANT_MODE) {
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
    TORCH_CHECK(
        gmm_weight_scale.defined(), "The input tensor gmm_weight_scale cannot be None.", OPS_ERROR(ErrCode::PARAM));
    int gmm_weight_scale_dim = (gmm_weight_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE;
    TORCH_CHECK(gmm_weight_scale.dim() == gmm_weight_scale_dim,
        "The dim of gmm_weight_scale should be ",
        gmm_weight_scale_dim,
        ", but got ",
        gmm_weight_scale.dim(),
        ".",
        OPS_ERROR(ErrCode::PARAM));
    if (gmm_weight_quant_mode_value == PERTENSOR_QUANT_MODE) {
        if (gmm_weight_scale_dtype.has_value()) {
            TORCH_CHECK(gmm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT),
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
    } else if (gmm_weight_quant_mode_value == MX_QUANT_MODE) {
        TORCH_CHECK(gmm_weight_scale_dtype.has_value(),
            "The input gmm_weight_scale_dtype must be provided for mx quant mode.",
            OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(gmm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E8M0),
            "The input gmm_weight_scale_dtype should be float8_e8m0 for mx quant mode, but got ",
            op_plugin::utils::DTypeToString(gmm_weight_scale_dtype.value()),
            ".",
            OPS_ERROR(ErrCode::VALUE));
    }
    // hcom
    char *hcom_ptr = const_cast<char *>(hcom.data());
    // ep_world_size
    TORCH_CHECK(SUPPORT_WORLD_SIZE_LIST.find(ep_world_size) != SUPPORT_WORLD_SIZE_LIST.end(),
        "The input ep_world_size should be in [2, 4, 8, 16, 32, 64, 128, 256], but got ",
        ep_world_size,
        ".",
        OPS_ERROR(ErrCode::VALUE));
    // send_counts_tensor
    const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
    // recv_counts_tensor
    const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
    // mm_x
    const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
    // mm_weight
    const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
    // mm_x_scale
    const at::Tensor &mm_x_scale_real = mm_x_scale.value_or(at::Tensor());
    // mm_weight_scale
    const at::Tensor &mm_weight_scale_real = mm_weight_scale.value_or(at::Tensor());
    // group_size
    TORCH_CHECK(!group_size.has_value(), "The input group_size should be None.", OPS_ERROR(ErrCode::PARAM));
    int64_t group_sizes = 0;
    // gmm_y
    int64_t n1 = gmm_weight.size(DIM_TWO);
    int64_t a = 0;
    for (auto &i : recv_counts) {
        a += i;
    }
    aclDataType gmm_y_acltype = c10_npu::GetAclDataType(gmm_y_dtype);
    at::ScalarType gmm_y_scalar_dtype = at_npu::native::OpPreparation::convert_to_scalar_type(gmm_y_acltype);
    auto gmm_y = at_npu::native::OpPreparation::apply_tensor_without_format({a, n1}, c10::dtype(gmm_y_scalar_dtype));
    // mm
    at::Tensor mm_y{nullptr};
    int64_t mm_x_quant_mode_value = NO_QUANT_MODE;
    int64_t mm_weight_quant_mode_value = NO_QUANT_MODE;

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
        mm_x_quant_mode_value = mm_x_quant_mode.value();
        mm_weight_quant_mode_value = mm_weight_quant_mode.value();
        const at::Tensor &mm_x_value = mm_x.value();
        const at::Tensor &mm_weight_value = mm_weight.value();
        // mm_x
        TORCH_CHECK(mm_x_value.dim() == DIM_TWO,
            "The dim of mm_x should be 2, but got ",
            mm_x_value.dim(),
            ".",
            OPS_ERROR(ErrCode::PARAM));
        if (mm_x_quant_mode_value == PERTENSOR_QUANT_MODE) {
            TORCH_CHECK(mm_x_dtype.has_value(),
                "The input mm_x_dtype must be provided for pertensor quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::HIFLOAT8),
                "The input mm_x_dtype should be hifloat8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(mm_x_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(
                mm_x_value.scalar_type() == at::ScalarType::Byte || mm_x_value.scalar_type() == at::ScalarType::Char,
                "The input mm_x tensor dtype should be uint8 or int8 for pertensor quant mode, but got ",
                op_plugin::utils::DTypeToString(static_cast<int64_t>(mm_x_value.scalar_type())),
                ".",
                OPS_ERROR(ErrCode::TYPE));
        } else if (mm_x_quant_mode_value == MX_QUANT_MODE) {
            TORCH_CHECK(mm_x_dtype.has_value(),
                "The input mm_x_dtype must be provided for mx quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                            mm_x_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2),
                "The input mm_x_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                op_plugin::utils::DTypeToString(mm_x_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
        }
        // mm_weight
        TORCH_CHECK(mm_weight_value.dim() == DIM_TWO,
            "The dim of mm_weight should be 2, but got ",
            mm_weight_value.dim(),
            ".",
            OPS_ERROR(ErrCode::PARAM));
        if (mm_weight_quant_mode_value == PERTENSOR_QUANT_MODE) {
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
        } else if (mm_weight_quant_mode_value == MX_QUANT_MODE) {
            TORCH_CHECK(mm_weight_dtype.has_value(),
                "The input mm_weight_dtype must be provided for mx quant mode.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(mm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E4M3FN) ||
                            mm_weight_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT8_E5M2),
                "The input mm_weight_dtype must be float8_e4m3_fn or float8_e5m2 for mx quant mode, but got ",
                op_plugin::utils::DTypeToString(mm_weight_dtype.value()),
                ".",
                OPS_ERROR(ErrCode::VALUE));
        }
        int64_t bs = mm_x_value.size(0);
        int64_t n2 = mm_weight_value.size(1);
        // mm_x_scale
        if (mm_x_scale.has_value()) {
            const at::Tensor &mm_x_scale_val = mm_x_scale.value();
            TORCH_CHECK(mm_x_scale_val.dim() == ((mm_x_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                "The dim of mm_x_scale should be ",
                ((mm_x_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                " for quant_mode ",
                mm_x_quant_mode_value,
                ", but got ",
                mm_x_scale_val.dim(),
                ".",
                OPS_ERROR(ErrCode::PARAM));

            if (mm_x_quant_mode_value == PERTENSOR_QUANT_MODE) {
                if (mm_x_scale_dtype.has_value()) {
                    TORCH_CHECK(mm_x_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT),
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
                mm_weight_scale_val.dim() == ((mm_weight_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                "The dim of mm_weight_scale should be ",
                ((mm_weight_quant_mode_value == MX_QUANT_MODE) ? DIM_THREE : DIM_ONE),
                " for quant_mode ",
                mm_weight_quant_mode_value,
                ", but got ",
                mm_weight_scale_val.dim(),
                ".",
                OPS_ERROR(ErrCode::PARAM));

            if (mm_weight_quant_mode_value == PERTENSOR_QUANT_MODE) {
                if (mm_weight_scale_dtype.has_value()) {
                    TORCH_CHECK(mm_weight_scale_dtype.value() == static_cast<int64_t>(c10_npu::DType::FLOAT),
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
            } else if (mm_weight_quant_mode_value == MX_QUANT_MODE) {
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
        aclDataType mm_y_acltype = c10_npu::GetAclDataType(mm_y_dtype.value());
        at::ScalarType mm_y_scalar_dtype = at_npu::native::OpPreparation::convert_to_scalar_type(mm_y_acltype);
        mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, c10::dtype(mm_y_scalar_dtype));
    }
    // permute_out
    at::Tensor permute_out{nullptr};
    if (permute_out_flag) {
        int64_t h1 = gmm_x.size(1);
        //  permute_out  shape
        aclDataType permute_out_acltype;
        if (gmm_x_dtype.has_value()) {
            permute_out_acltype = c10_npu::GetAclDataType(gmm_x_dtype.value());
        } else {
            permute_out_acltype = at_npu::native::OpPreparation::convert_to_acl_data_type(gmm_x.scalar_type());
        }
        at::ScalarType permute_out_scalar_dtype =
            at_npu::native::OpPreparation::convert_to_scalar_type(permute_out_acltype);
        permute_out =
            at_npu::native::OpPreparation::apply_tensor_without_format({a, h1}, c10::dtype(permute_out_scalar_dtype));
    }

    // aclnn 
    TensorWrapper gmm_x_wrapper = make_wrapper(gmm_x, gmm_x_dtype);
    TensorWrapper gmm_weight_wrapper = make_wrapper(gmm_weight, gmm_weight_dtype);
    TensorWrapper gmm_x_scale_wrapper = make_wrapper(gmm_x_scale, gmm_x_scale_dtype);
    TensorWrapper gmm_weight_scale_wrapper = make_wrapper(gmm_weight_scale, gmm_weight_scale_dtype);
    TensorWrapper mm_x_wrapper = make_wrapper(mm_x_real, mm_x_dtype);
    TensorWrapper mm_weight_wrapper = make_wrapper(mm_weight_real, mm_weight_dtype);
    TensorWrapper mm_x_scale_wrapper = make_wrapper(mm_x_scale_real, mm_x_scale_dtype);
    TensorWrapper mm_weight_scale_wrapper = make_wrapper(mm_weight_scale_real, mm_weight_scale_dtype);
    TensorWrapper permute_out_wrapper = {permute_out,
        (gmm_x_dtype.has_value()) ? c10_npu::GetAclDataType(gmm_x_dtype.value())
                                  : at_npu::native::OpPreparation::convert_to_acl_data_type(gmm_x.scalar_type())};
    bool transposeGmmWeight = false;
    bool transposeMmWeight = false;

    EXEC_NPU_CMD(aclnnAlltoAllvQuantGroupedMatMul,
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
        gmm_x_quant_mode_value,
        gmm_weight_quant_mode_value,
        mm_x_quant_mode_value,
        mm_weight_quant_mode_value,
        hcom_ptr,
        ep_world_size,
        send_counts,
        recv_counts,
        transposeGmmWeight,
        transposeMmWeight,
        group_sizes,
        permute_out_flag,
        gmm_y,
        mm_y,
        permute_out_wrapper);

    return std::tie(gmm_y, mm_y, permute_out);
}
}  // namespace op_api
