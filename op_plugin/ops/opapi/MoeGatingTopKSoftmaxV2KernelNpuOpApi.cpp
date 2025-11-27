// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at related link.
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

const int DIM_TWO = 2;
const int DIM_THREE = 3;

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_gating_top_k_softmax_v2 (const at::Tensor &x, int64_t k, const c10::optional<at::Tensor> &finished_opt,
        const c10::optional<int64_t> renorm_opt, const c10::optional<bool> softmax_flag_opt)
    {
        // check x's shape
        TORCH_CHECK(x.dim() == DIM_TWO or x.dim() == DIM_THREE, "The x's shape should be 2D or 3D", OPS_ERROR(ErrCode::PARAM));
        // check x's datatype
        TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
            "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ",
            x.scalar_type(), OPS_ERROR(ErrCode::PARAM));
        
        // check k's datatype
        auto x_size = x.sizes();
        TORCH_CHECK(k >= 0 and k <= x_size[x.dim() - 1], "The k's shape should be in [0, ", x_size[x.dim() - 1], "]", OPS_ERROR(ErrCode::PARAM));
        
        // renorm optional
        int64_t renorm = c10::value_or_else(renorm_opt, [] {return 0;});
        TORCH_CHECK(renorm == 0 || renorm == 1, "renorm must be 0 or 1, but got: ", renorm, OPS_ERROR(ErrCode::PARAM));
        
        bool softmax_result_flag = c10::value_or_else(softmax_flag_opt, [] {return false; });

        // finished Tensor
        const at::Tensor &finished = c10::value_or_else(finished_opt, [] {return at::Tensor(); });
        if (finished.defined()) {
            TORCH_CHECK(finished.scalar_type() == at::kBool, "bool tensor expected but got a tensor with dtype: ", finished.scalar_type(), OPS_ERROR(ErrCode::PARAM));
            auto finished_size = finished.sizes();
            TORCH_CHECK((x.dim() - 1) == finished.dim(), "x.dim() should be 1 more than finished.dim().", OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(x_size[0] == finished_size[0], "Input rows should be same as finished rows.", OPS_ERROR(ErrCode::PARAM));
            if (x.dim() == DIM_THREE) {
                TORCH_CHECK(x_size[1] == finished_size[1], "Input rows should be same as finished rows.", OPS_ERROR(ErrCode::PARAM));
            }
        }

        at::Tensor y;
        at::Tensor expert_idx;
        at::Tensor softmax_result;  // Optional output

        if (x.dim() == DIM_THREE) {
            y = npu_preparation::apply_tensor_without_format({x_size[0], x_size[1], k}, x.options());
            expert_idx = npu_preparation::apply_tensor_without_format(
                {x_size[0], x_size[1], k}, x.options().dtype(at::kInt));
        } else {
            y = npu_preparation::apply_tensor_without_format({x_size[0], k}, x.options());
            expert_idx = npu_preparation::apply_tensor_without_format(
                {x_size[0], k}, x.options().dtype(at::kInt));
        }

        bool softmaxFlag = (renorm == 0) && softmax_result_flag;
        if (softmaxFlag) {
            if (x.dim() == DIM_THREE) {
                softmax_result = npu_preparation::apply_tensor_without_format(
                    {x_size[0], x_size[1], x_size[2]}, x.options().dtype(at::kFloat));
            } else {
                softmax_result = npu_preparation::apply_tensor_without_format(
                    {x_size[0], x_size[1]}, x.options().dtype(at::kFloat));
            }
        } else {
            softmax_result = npu_preparation::apply_tensor_without_format({0}, x.options().dtype(at::kFloat));
        }

        if (k == 0) {
            return std::tie(y, expert_idx, softmax_result);
        }

        for (int32_t i = 0; i < x.dim(); i++) {
            if (x_size[i] == 0) {
                return std::tie(y, expert_idx, softmax_result);
            }
        }

        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmaxV2, x, finished, k, renorm, softmaxFlag, y, expert_idx, softmax_result);

        return std::tie(y, expert_idx, softmax_result);
    }
}