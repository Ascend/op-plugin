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

#include <ATen/WrapDimUtilsMulti.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
c10::SmallVector<int64_t, SIZE> logsumexp_npu_output_size(const at::Tensor &self, at::IntArrayRef dims, bool keepdim)
{
    return op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
}

at::Tensor squeeze_multiple(const at::Tensor &self, at::IntArrayRef dims)
{
    int ndims = static_cast<int>(self.sizes().size());
    auto dims_to_squeeze = at::dim_list_to_bitset(dims, ndims);
    at::Tensor result = self;
    for (int i = ndims - 1; i >= 0; --i) {
        if (dims_to_squeeze[i]) {
            result = result.squeeze(i);
        }
    }
    return result;
}

at::Tensor &logsumexp_out_nocheck(at::Tensor &result, const at::Tensor &self, at::IntArrayRef dims, bool keepdim)
{
    at::NoNamesGuard guard;
    if (self.numel() != 0) {
        at_npu::native::OpCommand cmd;
        auto maxes = acl_op::amax(self, dims, true);
        auto maxes_squeezed = (keepdim ? maxes : squeeze_multiple(maxes, dims));
        maxes_squeezed.masked_fill_(maxes_squeezed.abs() == INFINITY, 0);
        cmd.Name("ReduceLogSumExp").Input(self.sub(maxes)).Input(dims).Output(result).Attr("keep_dims", keepdim).Run();
        result.add_(maxes_squeezed);
    } else {
        at::sum_out(result, at::exp(self), dims, keepdim);
        result.log_();
    }
    at::namedinference::propagate_names_for_reduction(result, self, dims, keepdim);
    return result;
}
} // namespace

at::Tensor &logsumexp_out(const at::Tensor &self, at::DimnameList dims, bool keepdim, at::Tensor &result)
{
    return logsumexp_out(self, dimnames_to_positions(self, dims), keepdim, result);
}

at::Tensor &logsumexp_out(const at::Tensor &self, at::IntArrayRef dims, bool keepdim, at::Tensor &result)
{
    auto output_size = logsumexp_npu_output_size(self, dims, keepdim);
    npu_preparation::CheckOut({self}, result, self, output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contig_tensor = npu_utils::format_contiguous(result);
        logsumexp_out_nocheck(contig_tensor, self, dims, keepdim);
        npu_utils::format_fresh_view(result, contig_tensor);
    } else {
        logsumexp_out_nocheck(result, self, dims, keepdim);
    }
    return result;
}

at::Tensor logsumexp(const at::Tensor &self, at::IntArrayRef dims, bool keepdim)
{
    auto output_size = logsumexp_npu_output_size(self, dims, keepdim);
    at::Tensor result = npu_preparation::apply_tensor(self, output_size);
    return logsumexp_out_nocheck(result, self, dims, keepdim);
}

at::Tensor logsumexp(const at::Tensor &self, at::DimnameList dims, bool keepdim)
{
    return acl_op::logsumexp(self, dimnames_to_positions(self, dims), keepdim);
}

} // namespace acl_op
