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

at::Tensor nll_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index)
{
    return std::get<0>(at::nll_loss_forward(self, target, weight, reduction, ignore_index));
}

at::Tensor& nll_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& out)
{
    at::Tensor total_weight = npu_preparation::apply_tensor({}, self.options(), self);
    return std::get<0>(at::nll_loss_forward_out(out, total_weight, self, target, weight, reduction, ignore_index));
}

at::Tensor nll_loss2d(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index)
{
    return std::get<0>(at::nll_loss2d_forward(self, target, weight, reduction, ignore_index));
}

at::Tensor& nll_loss2d_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& out)
{
    at::Tensor total_weight = npu_preparation::apply_tensor({}, self.options(), self);
    return std::get<0>(at::nll_loss2d_forward_out(out, total_weight, self, target, weight, reduction, ignore_index));
}

at::Tensor& multilabel_margin_loss_out(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    at::Tensor& out)
{
    at::Tensor is_target = npu_preparation::apply_tensor(target);
    return std::get<0>(at::multilabel_margin_loss_forward_out(out, is_target, self, target, reduction));
}

at::Tensor multilabel_margin_loss(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction)
{
    return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}
} // namespace acl_op
