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

at::Tensor obfuscation_calculate(
    const at::Tensor &fd, const at::Tensor &x,
    const at::Tensor &param, c10::optional<double> obf_coefficient
    )
{
    int32_t fd_real;
    int32_t param_real;
    if (!fd.defined()) {
        throw std::runtime_error("fd cannot be empty");
    }
    try {
        auto fd_scalar = fd[0].item();
        fd_real = fd_scalar.to<int32_t>();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to process fd: " + std::string(e.what()));
    }
    if (!param.defined()) {
        throw std::runtime_error("param cannot be empty");
    }
    try {
        auto param_scalar = param[0].item();
        param_real = param_scalar.to<int32_t>();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to process param: " + std::string(e.what()));
    }
    auto cmd_real = static_cast<int32_t>(1);
    auto obf_coefficient_real = static_cast<float>(obf_coefficient.value_or(1));
    auto out_size = op_infer::array_to_small_vector(x.sizes());
    auto out_type = x.scalar_type();
    c10::TensorOptions options = x.options().dtype(out_type);
    at::Tensor y = npu_preparation::apply_tensor_without_format(out_size, options);
    EXEC_NPU_CMD(aclnnObfuscationCalculateV2, fd_real, x, param_real, cmd_real, obf_coefficient_real, y);
    return y;
}
}
