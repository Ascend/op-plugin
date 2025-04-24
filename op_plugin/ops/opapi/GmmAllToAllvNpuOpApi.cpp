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

namespace op_api {
    std::tuple<at::Tensor, at::Tensor> npu_gmm_alltoallv(const at::Tensor &gmm_x, const at::Tensor &gmm_weight,
                                                         c10::string_view hcom, int64_t ep_world_size, at::IntArrayRef send_counts, at::IntArrayRef recv_counts,
                                                         const c10::optional<at::Tensor> &send_counts_tensor,
                                                         const c10::optional<at::Tensor> &recv_counts_tensor, const c10::optional<at::Tensor> &mm_x,
                                                         const c10::optional<at::Tensor> &mm_weight, bool trans_gmm_weight, bool trans_mm_weight)
    {
        at::Tensor mm_y{nullptr};
        int64_t bsk = 0;
        for (auto &i : recv_counts) {
            bsk += i;
        }
        int64_t n1 = trans_gmm_weight ? gmm_weight.size(1) : gmm_weight.size(2);
        if (mm_x.has_value() && mm_weight.has_value()) {
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1);
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mm_x_value.options());
        }
        auto y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, gmm_x.options());
        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        const char *hcom_ptr = (char *)hcom.data();
        EXEC_NPU_CMD(aclnnGroupedMatMulAlltoAllv,
                     gmm_x,
                     gmm_weight,
                     send_count_tensor_real,
                     recv_count_tensor_real,
                     mm_x_real,
                     mm_weight_real,
                     hcom_ptr,
                     ep_world_size,
                     send_counts,
                     recv_counts,
                     trans_gmm_weight,
                     trans_mm_weight,
                     y,
                     mm_y);
        return std::tie(y, mm_y);
    }
} // namespace op_api
