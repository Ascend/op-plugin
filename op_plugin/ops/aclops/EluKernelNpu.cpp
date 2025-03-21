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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& elu_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale)
{
    float alpha_value = op_plugin::utils::get_scalar_float_value(alpha);
    float scale_value = op_plugin::utils::get_scalar_float_value(scale);
    float input_scale_value = op_plugin::utils::get_scalar_float_value(input_scale);
    at_npu::native::OpCommand cmd;
    cmd.Name("Elu")
        .Input(self)
        .Output(result)
        .Attr("alpha", alpha_value)
        .Attr("scale", scale_value)
        .Attr("input_scale", input_scale_value)
        .Run();
    return result;
}

at::Tensor elu_out_nocheck(
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    elu_out_nocheck(result, self, alpha, scale, input_scale);
    return result;
}
} // namespace

at::Tensor& elu_out(
    const at::Tensor& self,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        elu_out_nocheck(contiguous_result, self, alpha, scale, input_scale);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        elu_out_nocheck(out, self, alpha, scale, input_scale);
    }
    return out;
}

at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale)
{
    return elu_out_nocheck(self, alpha, scale, input_scale);
}

at::Tensor& elu_(at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale, const at::Scalar& input_scale)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        at::Tensor result = elu_out_nocheck(contiguous_self, alpha, scale, input_scale);
        npu_utils::format_fresh_view(self, result);
    } else {
        auto result = elu_out_nocheck(self, alpha, scale, input_scale);
        self.copy_(result);
    }
    return self;
}
} // namespace acl_op
