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
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

at::Tensor& fill_diagonal_out_npu(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Scalar& value,
    bool wrap)
{
    float fill_value = op_plugin::utils::get_scalar_float_value(value);
    at_npu::native::OpCommand cmd;
    cmd.Name("FillDiagonal")
        .Input(self)
        .Output(result)
        .Attr("fill_value", fill_value)
        .Attr("wrap", wrap)
        .Run();

    return result;
}

at::Tensor& fill_diagonal_(at::Tensor& self, const at::Scalar& fill_value, bool wrap)
{
    npu_preparation::CastBackToOriFormat(self);

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        fill_diagonal_out_npu(contiguous_self, contiguous_self, fill_value, wrap);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        fill_diagonal_out_npu(self, self, fill_value, wrap);
    }

    return self;
}

} // namespace acl_op
