// Copyright (c) 2024 Huawei Technologies Co., Ltd
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
c10::SmallVector<int64_t, SIZE> get_rstd_shape(const at::Tensor &self, const at::Tensor &gamma)
{
    c10::SmallVector<int64_t, SIZE> ret;
    auto rstd_dim = self.dim() - gamma.dim();
    for (int64_t i = 0; i < self.dim(); i++) {
        if (i < rstd_dim) {
            ret.emplace_back(self.size(i));
        } else {
            ret.emplace_back(1);
        }
    }
    return ret;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_add_rms_norm(
    const at::Tensor &x1,
    const at::Tensor &x2,
    const at::Tensor &gamma,
    double epsilon)
{
    TORCH_CHECK(x1.dim() >= gamma.dim(), "The gamma shape should not be bigger than self shape."
        + OPS_ERROR(ErrCode::PARAM));
    at::Tensor y = npu_preparation::apply_tensor(x1.sizes(), x1.options().dtype(gamma.dtype()), x1);
    auto rstd_shape = get_rstd_shape(x1, gamma);
    at::Tensor rstd = npu_preparation::apply_tensor(rstd_shape, x1.options().dtype(at::kFloat), x1);
    at::Tensor x = npu_preparation::apply_tensor(x1.sizes(), x1.options().dtype(gamma.dtype()), x1);

    at_npu::native::OpCommand cmd;
    cmd.Name("AddRmsNorm")
        .Input(x1, "x1")
        .Input(x2, "x2")
        .Input(gamma, "gamma")
        .Output(y, "y")
        .Output(rstd, "rstd")
        .Output(x, "x")
        .Attr("epsilon", static_cast<float>(epsilon))
        .Run();

    return std::make_tuple(y, rstd, x);
}
} // namespace acl_op
