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

at::Tensor obfuscation_finalize(const at::Tensor &fd_to_close)
{
    auto out_type = fd_to_close.scalar_type();
    auto out_size = op_infer::array_to_small_vector(fd_to_close.sizes());
    c10::TensorOptions options = fd_to_close.options().dtype(out_type);
    at::Tensor fd = npu_preparation::apply_tensor_without_format(out_size, options);
    auto fd_to_close_scalar = fd_to_close[0].item();
    auto fd_to_close_real = fd_to_close_scalar.to<int32_t>();
    int32_t data_type = 0;
    int32_t hidden_size = 0;
    int32_t tp_rank = 0;
    int32_t model_obf_seed_id = 0;
    int32_t data_obf_seed_id = 0;
    int32_t mode = 16;
    int32_t thread_num = 0;
    float obf_cft = 1.0;
    EXEC_NPU_CMD(aclnnObfuscationSetupV2, fd_to_close_real, data_type, hidden_size, tp_rank, model_obf_seed_id,
                 data_obf_seed_id, mode, thread_num, obf_cft, fd);
    return fd;
}
}
