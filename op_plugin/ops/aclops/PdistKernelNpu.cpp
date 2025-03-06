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

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& pdist_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    float p)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Pdist")
        .Input(self)
        .Attr("p", p)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor _pdist_forward(const at::Tensor& self, double p)
{
    at::Tensor result;
    if (self.size(0) <= 1) {
        result = npu_preparation::apply_tensor(self, {0});
    } else {
        // double is not supported in NPU,  type of P needs to be converted from double to float.
        float p_float;
        if (std::isinf(p)) {
            p_float = std::numeric_limits<float>::infinity();
        } else {
            TORCH_CHECK(
                p <= std::numeric_limits<float>::max(), "npu dose not support float64" + OPS_ERROR(ErrCode::TYPE));
            p_float = static_cast<float>(p);
        }
        auto output_size = op_infer::pdist_npu_output_size(self);
        result = npu_preparation::apply_tensor(self, output_size);
        if (self.size(1) == 0) {
            acl_op::fill_(result, 0);
        } else {
            pdist_out_npu_nocheck(result, self, p_float);
        }
    }
    return result;
}

at::Tensor pdist(const at::Tensor& self, double p)
{
    TORCH_CHECK(self.dim() == 2,
                "pdist only supports 2D tensors, got: ", self.dim(), "D", OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(at::isFloatingType(
        self.scalar_type()), "pdist only supports floating-point dtypes" + OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(p >= 0, "pdist only supports non-negative p values" + OPS_ERROR(ErrCode::VALUE));

    return at::_pdist_forward(self, p);
}
} // namespace acl_op
