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
                                                         const c10::optional<at::Tensor> &mm_weight, bool trans_gmm_weight, bool trans_mm_weight,
                                                         const c10::optional<at::Tensor> &global_token_per_expert,
                                                         const c10::optional<at::Tensor> &gmm_weight_scale,
                                                         const c10::optional<at::Tensor> &gmm_x_scale, c10::string_view comm_mode, int64_t y_dtype, int64_t output_token_num)
    {
        at::Tensor mm_y{nullptr};
        int64_t bsk = 0;
        int64_t default_scale = 2;
        int64_t dtype_float16 = 0;
        int64_t dtype_bfloat16 = 1;
        output_token_num = output_token_num == -1 ? default_scale * gmm_x.size(0) : output_token_num;
        if (comm_mode == "aiv") {
            bsk = output_token_num;
        } else {
            for (auto &i : recv_counts) {
                bsk += i;
            }
        }
        TORCH_CHECK((y_dtype == 0 || y_dtype == 1), "Invalid y_dtype value:", y_dtype,
                    ". Expected 0(fp16) or 1 (bfp16).");
        int64_t n1 = trans_gmm_weight ? gmm_weight.size(1) : gmm_weight.size(2);
        if (mm_x.has_value() && mm_weight.has_value()) {
            const at::Tensor &mm_x_value = mm_x.value();
            const at::Tensor &mm_weight_value = mm_weight.value();
            int64_t bs = mm_x_value.size(0);
            int64_t n2 = trans_mm_weight ? mm_weight_value.size(0) : mm_weight_value.size(1);
            auto mmy_options = mm_x_value.options();
            if (y_dtype == dtype_float16 && gmm_x.scalar_type() == at::kChar) {
                mmy_options = mmy_options.dtype(at::kHalf);
            } else if (y_dtype == dtype_bfloat16 && gmm_x.scalar_type() == at::kChar) {
                mmy_options = mmy_options.dtype(at::kBFloat16);
            }
            mm_y = at_npu::native::OpPreparation::apply_tensor_without_format({bs, n2}, mmy_options);
        }
        auto y_options = gmm_x.options();
        if (y_dtype == dtype_float16 && gmm_x.scalar_type() == at::kChar) {
            y_options = y_options.dtype(at::kHalf);
        } else if (y_dtype == dtype_bfloat16 && gmm_x.scalar_type() == at::kChar) {
            y_options = y_options.dtype(at::kBFloat16);
        }
        auto y = at_npu::native::OpPreparation::apply_tensor_without_format({bsk, n1}, y_options);
        const at::Tensor &mm_x_real = mm_x.value_or(at::Tensor());
        const at::Tensor &mm_weight_real = mm_weight.value_or(at::Tensor());
        const at::Tensor &send_count_tensor_real = send_counts_tensor.value_or(at::Tensor());
        const at::Tensor &recv_count_tensor_real = recv_counts_tensor.value_or(at::Tensor());
        at::IntArrayRef inplace_vec{0, 0};
        at::IntArrayRef send_counts_real = send_counts.empty() ? inplace_vec : send_counts;
        at::IntArrayRef recv_counts_real = recv_counts.empty() ? inplace_vec : recv_counts;
        char *hcom_ptr = const_cast<char *>(hcom.data());
        char *comm_mode_ptr = const_cast<char *>(comm_mode.data());
        at::Tensor empty_tensor = at::Tensor();
        auto global_token_per_expert_real = global_token_per_expert.value_or(at::Tensor());
        auto gmm_weight_scale_real = gmm_weight_scale.value_or(at::Tensor());
        auto gmm_x_scale_real = gmm_x_scale.value_or(at::Tensor());
        if (check_aclnn_kernel_available("aclnnGroupedMatMulAlltoAllvV2")) {
            EXEC_NPU_CMD(aclnnGroupedMatMulAlltoAllvV2, gmm_x, gmm_weight, send_count_tensor_real,
                         recv_count_tensor_real, mm_x_real, mm_weight_real, global_token_per_expert_real,
                         gmm_x_scale_real, gmm_weight_scale_real, hcom_ptr, ep_world_size, send_counts_real,
                         recv_counts_real, trans_gmm_weight, trans_mm_weight, comm_mode_ptr, y, mm_y);
        } else {
            EXEC_NPU_CMD(aclnnGroupedMatMulAlltoAllv, gmm_x, gmm_weight, send_count_tensor_real, recv_count_tensor_real,
                         mm_x_real, mm_weight_real, hcom_ptr, ep_world_size, send_counts, recv_counts, trans_gmm_weight,
                         trans_mm_weight, y, mm_y);
        }
        return std::tie(y, mm_y);
    }
} // namespace op_api
