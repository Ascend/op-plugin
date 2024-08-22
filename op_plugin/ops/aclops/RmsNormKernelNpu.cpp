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
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

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

std::tuple<at::Tensor, at::Tensor> npu_rms_norm(const at::Tensor &self, const at::Tensor &gamma, double epsilon)
{
    TORCH_CHECK(self.dim() >= gamma.dim(), "The gamma shape should not be bigger than self shape." + OPS_ERROR(ErrCode::PARAM));
    at::Tensor y = npu_preparation::apply_tensor(self.sizes(), self.options(), self);
    auto rstd_shape = get_rstd_shape(self, gamma);
    at::Tensor rstd = npu_preparation::apply_tensor(rstd_shape, self.options().dtype(at::kFloat), self);

    at_npu::native::OpCommand cmd;
    cmd.Name("RmsNorm")
        .Input(self, "x")
        .Input(gamma, "gamma")
        .Output(y, "y")
        .Output(rstd, "rstd")
        .Attr("epsilon", static_cast<float>(epsilon))
        .Run();

    return std::make_tuple(y, rstd);
}
} // namespace acl_op
