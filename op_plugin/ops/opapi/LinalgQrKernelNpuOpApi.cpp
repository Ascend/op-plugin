// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
using small_vector_list = std::tuple<c10::SmallVector<int64_t, op_infer::N>, c10::SmallVector<int64_t, op_infer::N>>;

static inline bool mode_valid(c10::string_view mode)
{
    return (mode == "reduced" || mode == "complete" || mode == "r");
}

static inline int64_t get_mode(c10::string_view mode)
{
    if (mode == "complete") {
        // complete模式对应输入为1
        return 1;
    }
    if (mode == "r") {
        // r模式对应输入为2
        return 2;
    }
    return 0;
}

static small_vector_list linalg_qr_infer_shape(const at::Tensor &self, c10::string_view mode)
{
    int m = self.size(-2);
    int n = self.size(-1);
    auto k = std::min<int>(m, n);
    auto shape = op_infer::array_to_small_vector(self.sizes());
    c10::SmallVector<int64_t, op_infer::N> Esize = {0};
    c10::SmallVector<int64_t, op_infer::N> Qsize(shape.begin(), shape.end() - 2);
    c10::SmallVector<int64_t, op_infer::N> Rsize(shape.begin(), shape.end() - 2);
    if (mode == "r") {
        Qsize = Esize;
        Rsize.insert(Rsize.end(), {k, n});
    } else if (mode == "complete") {
        Qsize.insert(Qsize.end(), {m, m});
        Rsize.insert(Rsize.end(), {m, n});
    } else {
        Qsize.insert(Qsize.end(), {m, k});
        Rsize.insert(Rsize.end(), {k, n});
    }
    return std::tie(Qsize, Rsize);
}

std::tuple<at::Tensor &, at::Tensor &> linalg_qr_out(const at::Tensor &self, c10::string_view mode, at::Tensor &Q,
                                                     at::Tensor &R)
{
    DO_COMPATIBILITY(aclnnLinalgQr, acl_op::linalg_qr_out(self, mode, Q, R));
    // 输入至少为2维tensor
    TORCH_CHECK(self.ndimension() >= 2, "Expected nonempty least 2D tensor, but got a tensor with sizes ",
                self.dim(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mode_valid(mode),
                "qr received unrecognized mode but expected one of 'reduced'(default), 'r', or 'complete'",
                OPS_ERROR(ErrCode::TYPE));
    auto sizes = linalg_qr_infer_shape(self, mode);
    npu_preparation::check_tensor({self}, Q, self, std::get<0>(sizes));
    npu_preparation::check_tensor({self}, R, self, std::get<1>(sizes));
    int64_t mode_int = get_mode(mode);
    EXEC_NPU_CMD(aclnnLinalgQr, self, mode_int, Q, R);
    return std::tie(Q, R);
}

std::tuple<at::Tensor, at::Tensor> linalg_qr(const at::Tensor &self, c10::string_view mode)
{
    DO_COMPATIBILITY(aclnnLinalgQr, acl_op::linalg_qr(self, mode));
    TORCH_CHECK(self.ndimension() >= 2, "Expected nonempty least 2D tensor, but got a tensor with sizes ",
                self.dim(), OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(mode_valid(mode), "Mode should be reduced, complete or r", OPS_ERROR(ErrCode::TYPE));
    auto sizes = linalg_qr_infer_shape(self, mode);
    at::Tensor Q = npu_preparation::apply_tensor_without_format(std::get<0>(sizes), self.options());
    at::Tensor R = npu_preparation::apply_tensor_without_format(std::get<1>(sizes), self.options());
    int64_t mode_int = get_mode(mode);
    EXEC_NPU_CMD(aclnnLinalgQr, self, mode_int, Q, R);
    return std::tie(Q, R);
}

} // namespace op_api
