// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
    using npu_preparation = at_npu::native::OpPreparation;
    using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

    tensor_list npu_moe_gating_top_k_softmax(const at::Tensor &x,
                                             const c10::optional<at::Tensor> &finished_opt,
                                             int64_t k)
    {
        TORCH_CHECK(x.dim() == 2 or x.dim() == 3, "The x should be 2D or 3D", OPS_ERROR(ErrCode::PARAM));
        TORCH_CHECK(
            x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16,
            "float16, float32 or bfloat16 tensor expected but got a tensor with dtype: ",
            x.scalar_type(), OPS_ERROR(ErrCode::PARAM));

        auto x_size = x.sizes();
        TORCH_CHECK(k >= 0 and k <= x_size[x.dim() - 1],
            "The k should be in [0, ", x_size[x.dim() - 1], "]", OPS_ERROR(ErrCode::PARAM));
        const at::Tensor &finished = c10::value_or_else(finished_opt, [] { return at::Tensor(); });
        if (finished.defined()) {
            TORCH_CHECK(
                finished.scalar_type() == at::kBool,
                "bool tensor expected but got a tensor with dtype: ",
                finished.scalar_type(), OPS_ERROR(ErrCode::PARAM));
            auto finished_size = finished.sizes();
            TORCH_CHECK((x.dim() - 1) == finished.dim(), "x dims shoud be largs finished dims than 1.",
                OPS_ERROR(ErrCode::PARAM));
            TORCH_CHECK(x_size[0] == finished_size[0], "Input rows shoud be same.", OPS_ERROR(ErrCode::PARAM));
            if (x.dim() == 3) {
                TORCH_CHECK(x_size[1] == finished_size[1], "Input rows shoud be same.", OPS_ERROR(ErrCode::PARAM));
            }
        }

        at::Tensor y;
        at::Tensor expert_idx;
        at::Tensor row_idx;
        if (x.dim() == 3) {
            y = npu_preparation::apply_tensor_without_format({x_size[0], x_size[1], k}, x.options());
            expert_idx = npu_preparation::apply_tensor_without_format({x_size[0], x_size[1], k},
                                                                      x.options().dtype(at::kInt));
            row_idx = npu_preparation::apply_tensor_without_format({x_size[0], x_size[1], k},
                                                                   x.options().dtype(at::kInt));
        } else {
            y = npu_preparation::apply_tensor_without_format({x_size[0], k}, x.options());
            expert_idx = npu_preparation::apply_tensor_without_format({x_size[0], k}, x.options().dtype(at::kInt));
            row_idx = npu_preparation::apply_tensor_without_format({x_size[0], k}, x.options().dtype(at::kInt));
        }

        if (k == 0) {
            return std::tie(y, expert_idx, row_idx);
        }

        for (int32_t i = 0; i < x.dim(); i++) {
            if (x_size[i] == 0) {
                return std::tie(y, expert_idx, row_idx);
            }
        }

        EXEC_NPU_CMD(aclnnMoeGatingTopKSoftmax, x, finished, k, y, expert_idx, row_idx);

        return std::tie(y, expert_idx, row_idx);
    }

}