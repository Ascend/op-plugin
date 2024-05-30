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

namespace {
at::Tensor &argsort_out_npu_nocheck(at::Tensor &values, at::Tensor &indices, const at::Tensor &self, int64_t dim,
                                    bool descending)
{
    at_npu::native::OpCommand cmd;
    if (values.dtype() == at::kInt || values.dtype() == at::kLong) {
        TORCH_NPU_WARN_ONCE("Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel "
                            "is running on AiCpu."
                            "If you are more concerned about high-performance execution,please cast dtype to float32.");
    }
    cmd.Name("Sort").Input(self).Output(values).Output(indices).Attr("axis", dim).Attr("descending", descending).Run();
    return indices;
}
} // namespace

at::Tensor argsort(const at::Tensor &self, int64_t dim, bool descending)
{
    dim = op_infer::make_wrap_dim(dim, self.dim());
    int64_t last_dim = op_infer::make_wrap_dim(-1, self.dim());

    at::SmallVector<int64_t, SIZE> perm;
    for (int64_t i = 0; i < self.dim(); i++) {
        perm.emplace_back(i);
    }
    std::swap(perm[dim], perm[last_dim]);

    auto output_size = op_infer::transpose_npu_output_size(self, perm);
    at::Tensor indices = npu_preparation::apply_tensor(self, self.options().dtype(at::kInt));
    at::Tensor transpose_self = acl_op::npu_transpose(self, perm, true);
    at::Tensor transpose_values = npu_preparation::apply_tensor(self, output_size);
    at::Tensor transpose_indices = npu_preparation::apply_tensor(indices, output_size);

    argsort_out_npu_nocheck(transpose_values, transpose_indices, transpose_self, last_dim, descending);
    acl_op::npu_transpose_out(transpose_indices, perm, true, indices);
    indices = at_npu::native::custom_ops::npu_dtype_cast(indices, at::kLong);
    return indices;
}

at::Tensor argsort(const at::Tensor &self, at::Dimname dim, bool descending)
{
    return acl_op::argsort(self, dimname_to_position(self, dim), descending);
}
} // namespace acl_op
