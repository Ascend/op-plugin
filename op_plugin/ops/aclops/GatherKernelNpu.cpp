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

#include <ATen/NamedTensorUtils.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& gather_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index)
{
    if (self.scalar_type() == at::kLong) {
        TORCH_NPU_WARN_ONCE("The oprator of gather is executed, Currently High Accuracy but Low Performance OP"
        "with 64-bit has been used,Please Do Some Cast at Python Functions with 32-bit for Better Performance!");
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("GatherElements")
        .Input(self)
        .Input(index)
        .Attr("dim", dim)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor& gather_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out)
{
    auto output_size = index.sizes();
    npu_preparation::CheckOut(
        {self},
        out,
        self,
        output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out);
        gather_out_npu_nocheck(contiguous_out, self, dim, index);
        npu_utils::format_fresh_view(out, contiguous_out);
    } else {
        gather_out_npu_nocheck(out, self, dim, index);
    }
    return out;
}

at::Tensor& gather_out(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad,
    at::Tensor& out)
{
    return acl_op::gather_out(self, dimname_to_position(self, dim), index, sparse_grad, out);
}

at::Tensor gather(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    bool sparse_grad)
{
    at::Tensor result = npu_preparation::apply_tensor(self, index.sizes());
    gather_out_npu_nocheck(result, self, dim, index);
    return result;
}

at::Tensor gather(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    bool sparse_grad)
{
    return acl_op::gather(self, dimname_to_position(self, dim), index, sparse_grad);
}
}  // op_plugin
