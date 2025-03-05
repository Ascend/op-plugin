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

#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
at::Tensor& softplus_backward_out_common_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar beta,
    at::Scalar threshold)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SoftplusV2Grad")
        .Input(grad_output)
        .Input(self)
        .Output(grad_input)
        .Attr("beta", beta)
        .Attr("threshold", threshold)
        .Run();

    return grad_input;
}
} // namespace acl_op
