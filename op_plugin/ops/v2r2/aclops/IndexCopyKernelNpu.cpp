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
#include <ATen/native/NonSymbolicBC.h>

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& index_copy_npu_impl(
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& result)
{
    index_copy_npu_par_check(dim, index, source, result);
    int64_t num_indices = index.numel();
    int64_t i;
    at::Tensor des;
    at::Tensor src;
    if (result.dim() > 1) {
        for (i = 0; i < num_indices; i++) {
            auto index_i = index.dim() == 0 ? index.item<int64_t>() : index[i].item<int64_t>();
            des = at::native::select(result, dim, index_i);
            src = at::native::select(source, dim, i);
            at_npu::native::NPUNativeFunctions::copy_(des, src, false);
        }
    } else {
        if (index.dim() == 0) {
            des = result.dim() == 0 ? result : result[index.item<int64_t>()];
            src = source.dim() == 0 ? source : source[0];
            at_npu::native::NPUNativeFunctions::copy_(des, src, false);
        } else {
            for (i = 0; i < num_indices; i++) {
                des = result.dim() == 0 ? result : result[index[i].item<int64_t>()];
                src = source.dim() == 0 ? source : source[i];
                at_npu::native::NPUNativeFunctions::copy_(des, src, false);
            }
        }
    }

    return result;
}
}  // namespace

at::Tensor index_copy(const at::Tensor& self, const int64_t dim, const at::Tensor& index, const at::Tensor& source)
{
    at::Tensor contiguous_self(self.clone());
    if (!npu_utils::check_match(&self)) {
        contiguous_self = npu_utils::format_contiguous(contiguous_self);
    }
    return index_copy_npu_impl(dim, index, source, contiguous_self);
}

at::Tensor& index_copy_(at::Tensor& self, const int64_t dim, const at::Tensor& index, const at::Tensor& source)
{
    at::Tensor contiguous_self(self);
    if (!npu_utils::check_match(&self)) {
        contiguous_self = npu_utils::format_contiguous(self);
    }
    at::Tensor result = index_copy_npu_impl(dim, index, source, contiguous_self);
    npu_utils::format_fresh_view(self, result);

    return self;
}
}  // namespace acl_op
