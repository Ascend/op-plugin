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
at::Tensor& dot_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& tensor)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Dot")
        .Input(self)
        .Input(tensor)
        .Output(result)
        .Run();

    return result;
}
} // namespace

at::Tensor& dot_out(const at::Tensor& self, const at::Tensor& tensor, at::Tensor& result)
{
    auto self_dtype = self.scalar_type();
    TORCH_CHECK(self_dtype != at::kInt && self_dtype != at::kByte && self_dtype != at::kChar,
        "'dot_npu' not implemented for 'Int'" + OPS_ERROR(ErrCode::TYPE));
    c10::SmallVector<int64_t, N> output_size = {};
    npu_preparation::CheckOut(
        {self, tensor},
        result,
        self,
        output_size);

    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        dot_out_npu_nocheck(contiguous_result, self, tensor);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        dot_out_npu_nocheck(result, self, tensor);
    }

    return result;
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor)
{
    auto self_dtype = self.scalar_type();
    TORCH_CHECK(self_dtype != at::kInt && self_dtype != at::kByte && self_dtype != at::kChar,
        "'dot_npu' not implemented for 'Int'" + OPS_ERROR(ErrCode::TYPE));
    c10::SmallVector<int64_t, N> output_size = {};
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    dot_out_npu_nocheck(result, self, tensor);
    return result;
}
} // acl_op
