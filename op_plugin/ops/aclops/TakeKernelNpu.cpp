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

namespace {
at::Tensor& take_out_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& index)
{
    at::Tensor input_tensor = self.reshape(-1);
    at::Tensor contiguous_self = npu_utils::format_contiguous(input_tensor);
    at::Tensor contiguous_index = npu_utils::format_contiguous(index);

    at_npu::native::OpCommand cmd;
    cmd.Name("Gather")
        .Input(contiguous_self)
        .Input(contiguous_index)
        .Output(result)
        .Attr("validate_indices", false)
        .Run();
    return result;
}
} // namespace

at::Tensor& take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& out)
{
    auto output_size = op_infer::input_same_output_size(index);

    npu_preparation::CheckOut(
        {self, index},
        out,
        self,
        output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        take_out_nocheck(contiguous_result, self, index);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        take_out_nocheck(out, self, index);
    }
    return out;
}

at::Tensor take(const at::Tensor& self, const at::Tensor& index)
{
    at::Tensor result = npu_preparation::apply_tensor(self, index.sizes());
    take_out_nocheck(result, self, index);
    return result;
}
} // namespace acl_op
