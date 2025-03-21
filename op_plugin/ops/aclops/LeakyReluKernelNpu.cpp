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
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& leaky_relu_out_nocheck(at::Tensor& result, const at::Tensor& self, at::Scalar negval)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("LeakyRelu")
        .Input(self)
        .Output(result)
        .Attr("negative_slope", negval)
        .Run();

    return result;
}
} // namespace

at::Tensor& leaky_relu_out(const at::Tensor& self, const at::Scalar& negative_slope, at::Tensor& out)
{
    npu_preparation::CheckOut(
        {self},
        out,
        self);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        leaky_relu_out_nocheck(contiguous_result, self, negative_slope);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        leaky_relu_out_nocheck(out, self, negative_slope);
    }
    return out;
}

at::Tensor leaky_relu(const at::Tensor& self, const at::Scalar& negative_slope)
{
    at::Tensor result = npu_preparation::apply_tensor(self);
    leaky_relu_out_nocheck(result, self, negative_slope);
    return result;
}

at::Tensor& leaky_relu_(at::Tensor& self, const at::Scalar& negative_slope)
{
    return acl_op::leaky_relu_out(self, negative_slope, self);
}
} // namespace acl_op
