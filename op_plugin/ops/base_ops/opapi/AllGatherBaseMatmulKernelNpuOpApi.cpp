// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_gather_mm(const at::Tensor &x1, const at::Tensor &x2,
                                                                           int64_t world_size, int64_t gather_index)
{
    auto out_x = gather_index == 0 ? x1.size(0) * world_size : x1.size(0);
    auto out_y = x2.size(1);
    return {out_x, out_y};
}

static c10::SmallVector<int64_t, op_infer::SIZE> get_output_size_gather(const at::Tensor &x1, const at::Tensor &x2,
                                                                        int64_t world_size, int64_t gather_index)
{
    const at::Tensor &gather_out = gather_index == 0 ? x1 : x2;
    return {gather_out.size(0) * world_size, gather_out.size(1)};
}

std::tuple<at::Tensor, at::Tensor> npu_all_gather_base_mm(const at::Tensor &self, const at::Tensor &x2,
                                                          c10::string_view hcom, int64_t world_size,
                                                          const c10::optional<at::Tensor> &bias, int64_t gather_index,
                                                          bool gather_output, int64_t comm_turn)
{
    TORCH_CHECK(self.dim() == 2 && x2.dim() == 2, "Both inputs of mm are required to be 2D, but the actual inputs are ",
                self.dim(), "D and ", x2.dim(), "D");
    TORCH_CHECK(self.size(1) == x2.size(0),
                "The K-axis in the two inputs of Matmul must be equal, but in reality, the K-axis of x1 is ",
                self.size(1), " and the K-axis of x2 is ", x2.size(0));
    auto out_gather_mm_size = get_output_size_gather_mm(self, x2, world_size, gather_index);
    auto out_gather_size = get_output_size_gather(self, x2, world_size, gather_index);
    auto out_gather_mm = at_npu::native::OpPreparation::apply_tensor_without_format(out_gather_mm_size, self.options());
    at::Tensor out_gather = at::empty({0}, self.options());
    if (gather_output) {
        out_gather = at_npu::native::OpPreparation::apply_tensor_without_format(out_gather_size, self.options());
    }
    const at::Tensor &bias_real = bias.value_or(at::Tensor());
    char *hcom_ptr = const_cast<char *>(hcom.data());
    int64_t stream_mode = ACL_STOP_ON_FAILURE;
    EXEC_NPU_CMD(aclnnAllGatherMatmul, self, x2, bias_real, hcom_ptr, gather_index, comm_turn, stream_mode,
                 out_gather_mm, out_gather);
    return std::tie(out_gather_mm, out_gather);
}
}  // namespace op_api
