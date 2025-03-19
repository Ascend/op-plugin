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
at::Tensor linalg_cross_dest_output(const at::Tensor &self, const at::Tensor &other)
{
    bool is_self_wrapped = npu_preparation::is_scalar_wrapped_to_tensor(self);
    return is_self_wrapped ? other : self;
}

at::Tensor &linalg_cross_out_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other,
                                     c10::optional<int64_t> dim)
{
    int64_t real_dim = dim.has_value() ? dim.value() : -65530;
    at_npu::native::OpCommand cmd;
    cmd.Name("Cross").Input(self).Input(other).Output(result).Attr("dim", real_dim).Run();
    return result;
}
} // namespace

at::Tensor &linalg_cross_out(const at::Tensor &self, const at::Tensor &other, const int64_t dim, at::Tensor &out)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor output_tensor = linalg_cross_dest_output(self, other);
    npu_preparation::CheckOut({self}, out, npu_preparation::get_tensor_npu_format(output_tensor), self.scalar_type(),
                              output_size);

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        linalg_cross_out_nocheck(contiguous_result, self, other, dim);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        linalg_cross_out_nocheck(out, self, other, dim);
    }
    return out;
}

at::Tensor linalg_cross(const at::Tensor &self, const at::Tensor &other, const int64_t dim)
{
    auto output_size = op_infer::broadcast_ops_npu_output_size(self, other);
    at::Tensor output_tensor = linalg_cross_dest_output(self, other);
    at::Tensor result = npu_preparation::apply_tensor(output_size, self.options(), output_tensor);
    linalg_cross_out_nocheck(result, self, other, dim);
    return result;
}
} // namespace acl_op
