// Copyright (c) 2025 Huawei Technologies Co., Ltd
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
enum class ObfDataType : int32_t {
    FLOAT = 0,
    HALF = 1,
    CHAR = 2,
    BFLOAT = 27,
    UNDEFINED = -1
};

at::Tensor obfuscation_initialize(
    int64_t hidden_size, int64_t tp_rank,
    int64_t cmd, c10::optional<c10::ScalarType> data_type,
    c10::optional<int64_t> model_obf_seed_id,
    c10::optional<int64_t> data_obf_seed_id,
    c10::optional<int64_t> thread_num,
    c10::optional<double> obf_coefficient
    )
{
    auto hidden_size_real = static_cast<int32_t>(hidden_size);
    auto tp_rank_real = static_cast<int32_t>(tp_rank);
    auto cmd_real = static_cast<int32_t>(cmd);
    int32_t data_type_real;
    if (!data_type.has_value()) {
        throw std::runtime_error("data_type cannot be empty");
    }

    switch (data_type.value()) {
        case at::ScalarType::Half:
            data_type_real = static_cast<int32_t>(ObfDataType::HALF);
            break;
        case at::ScalarType::Float:
            data_type_real = static_cast<int32_t>(ObfDataType::FLOAT);
            break;
        case at::ScalarType::Char:
            data_type_real = static_cast<int32_t>(ObfDataType::CHAR);
            break;
        case at::ScalarType::BFloat16:
            data_type_real = static_cast<int32_t>(ObfDataType::BFLOAT);
            break;
        default:
            data_type_real = static_cast<int32_t>(ObfDataType::UNDEFINED);
    }
    auto model_obf_seed_id_real = static_cast<int32_t>(model_obf_seed_id.value_or(0));
    auto data_obf_seed_id_real = static_cast<int32_t>(data_obf_seed_id.value_or(0));
    auto thread_num_real = static_cast<int32_t>(thread_num.value_or(4));
    auto obf_coefficient_real = static_cast<float>(obf_coefficient.value_or(1));
    at::ScalarType out_type = at::ScalarType::Int;
    c10::SmallVector<int64_t> out_size = {1};
    auto options = c10::TensorOptions().device(c10::DeviceType::PrivateUse1).dtype(out_type);
    int32_t fd_to_close = 0;
    at::Tensor fd = npu_preparation::apply_tensor_without_format(out_size, options);
    EXEC_NPU_CMD(aclnnObfuscationSetupV2, fd_to_close, data_type_real, hidden_size_real, tp_rank_real, model_obf_seed_id_real,
                 data_obf_seed_id_real, cmd_real, thread_num_real, obf_coefficient_real, fd);
    return fd;
}
}
