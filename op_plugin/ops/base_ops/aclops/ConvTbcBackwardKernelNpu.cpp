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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_tbc_backward(const at::Tensor &self, const at::Tensor &input,
                                                                 const at::Tensor &weight, const at::Tensor &bias,
                                                                 int64_t pad)
{
    TORCH_CHECK(input.dim() >= 3, "input has to be more than 3D, but got Tensor of dimension ", input.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(self.dim() >= 3, "self has to be more than 3D, but got Tensor of dimension ", self.dim(),
        OPS_ERROR(ErrCode::PARAM));
    TORCH_CHECK(weight.dim() >= 3, "weight has to be more than 3D, but got Tensor of dimension ", weight.dim(),
        OPS_ERROR(ErrCode::PARAM));
    auto output =
        acl_op::npu_conv2d_backward(input.permute({1, 2, 0}).unsqueeze(2), self.permute({1, 2, 0}).unsqueeze(2),
                                    weight.permute({2, 1, 0}).unsqueeze(2), {1, 1}, {0, pad}, {1, 1}, 1, {1, 1, 1});

    return std::make_tuple(std::move((std::get<0>(output)).squeeze(2).permute({2, 0, 1})),
                           std::move((std::get<1>(output)).squeeze(2).permute({2, 1, 0})),
                           std::move(std::get<2>(output)));
}
} // namespace acl_op
