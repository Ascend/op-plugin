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

#include <ATen/record_function.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor &npu_transpose_out_nocheck(at::Tensor &result, const at::Tensor &self, at::IntArrayRef perm,
                                      bool require_contiguous)
{
    at_npu::native::OpCommand cmd;
    if (require_contiguous) {
        // Any tensor-view(discontiguous) Input Tensor from users should be transformed to be contiguous here.
        cmd.Name("Transpose").Input(self).Input(perm).Output(result).Run();
    } else {
        // For permute-opt in trans-contiguous, it accepts transposed(discontiguous) Input Tensor.
        cmd.Name("Transpose").InputWithoutContiguous(self).Input(perm).Output(result).Run();
    }
    return result;
}
} // namespace

at::Tensor npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous)
{
    auto output_size = op_infer::transpose_npu_output_size(self, perm);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    npu_transpose_out_nocheck(result, self, perm, require_contiguous);

    return result;
}

at::Tensor &npu_transpose_out(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous, at::Tensor &result)
{
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        npu_transpose_out_nocheck(contiguous_result, self, perm, require_contiguous);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        npu_transpose_out_nocheck(result, self, perm, require_contiguous);
    }
    return result;
}

at::Tensor &npu_transpose_trans_contiguous_out(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous,
                                               at::Tensor &result)
{
    npu_transpose_out_nocheck(result, self, perm, require_contiguous);
    return result;
}

} // namespace acl_op
