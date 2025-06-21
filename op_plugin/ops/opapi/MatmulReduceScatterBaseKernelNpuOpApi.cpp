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

at::Tensor npu_mm_reduce_scatter_base(const at::Tensor &self, const at::Tensor &x2, c10::string_view hcom,
                                      int64_t world_size, c10::string_view reduce_op,
                                      const c10::optional<at::Tensor> &bias, int64_t comm_turn)
{
    TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8 || world_size == 16 || world_size == 32,
                "world_size should be in [2, 4, 8, 16, 32], but the actual value is ", world_size, "." + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(self.dim() == 2 && x2.dim() == 2, "Both inputs of mm are required to be 2D, but the actual inputs are ",
                self.dim(), "D and ", x2.dim(), "D." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(1) == x2.size(0),
                "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x1 is ",
                self.size(1), " and the K-axis of x2 is ", x2.size(0), "." + OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.size(0) % world_size == 0, "The M-axis in input of Matmul should be be divisible by world_size."
                + OPS_ERROR(ErrCode::PARAM));
    auto output_size = {self.size(0) / world_size, x2.size(1)};
    auto result = at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options());
    char *reduce_op_ptr = const_cast<char *>(reduce_op.data());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    EXEC_NPU_CMD(aclnnMatmulReduceScatter, self, x2, bias_real, hcom_ptr, reduce_op_ptr, comm_turn, stream_mode,
                 result);

    FLOP_COUNT(FlopCounter::mm_flop, self, x2);
    return result;
}
}
