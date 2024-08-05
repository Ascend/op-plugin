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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"
#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
#include <ATen/native/NonSymbolicBC.h>
#endif

namespace acl_op {
using npu_utils = at_npu::native::NpuUtils;

namespace {
#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor &index_copy_npu_impl(const int64_t dim, const at::Tensor &index, const at::Tensor &source,
                                at::Tensor &result)
{
    index_copy_npu_par_check(dim, index, source, result);
    int64_t num_indices = index.numel();
    int64_t i;
    if (result.dim() > 1) {
        at::Tensor des;
        at::Tensor src;
        for (i = 0; i < num_indices; i++) {
            des = at::native::select(result, dim, index[i].item<int64_t>());
            src = at::native::select(source, dim, i);
            at_npu::native::NPUNativeFunctions::copy_(des, src, false);
        }
    } else {
        for (i = 0; i < num_indices; i++) {
            auto idx = index[i].item<int64_t>();
            result[idx] = source[i];
        }
    }
    return result;
}


at::Tensor index_copy_npu(const at::Tensor &self, const int64_t dim, const at::Tensor &index, const at::Tensor &source)
{
    at::Tensor result(self.clone());
    return index_copy_npu_impl(dim, index, source, result);
}

at::Tensor index_copy_npu(const at::Tensor &self, const at::Dimname dim, const at::Tensor &index,
                          const at::Tensor &source)
{
    at::Tensor result(self.clone());
    return index_copy_npu_impl(dimname_to_position(self, dim), index, source, result);
}

at::Tensor &index_copy_npu_(at::Tensor &self, const at::Dimname dim, const at::Tensor &index, const at::Tensor &source)
{
    at::Tensor contiguous_self(self);
    if (!npu_utils::check_match(&self)) {
        contiguous_self = npu_utils::format_contiguous(self);
    }
    at::Tensor result = index_copy_npu_impl(dimname_to_position(self, dim), index, source, contiguous_self);
    npu_utils::format_fresh_view(self, result);

    return self;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
at::Tensor& index_copy_npu_impl(
    const int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& result)
{
    index_copy_npu_par_check(dim, index, source, result);
    int64_t num_indices = index.numel();
    int64_t i;
    if (result.dim() > 1) {
        at::Tensor des;
        at::Tensor src;
        for (i = 0; i < num_indices; i++) {
            des = at::native::select(result, dim, index[i].item<int64_t>());
            src = at::native::select(source, dim, i);
            at_npu::native::NPUNativeFunctions::copy_(des, src, false);
        }
    } else if (index.dim() == 0) {
        result[index.item<int64_t>()] = source[0];
    } else {
        for (i = 0; i < num_indices; i++) {
            result[index[i].item<int64_t>()] = source[i];
        }
    }
    return result;
}
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
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
#endif
} // namespace

#if VERSION_BETWEEN(V1R11, V1R11) || VERSION_BETWEEN(V2R0, V2R0)
at::Tensor index_copy(const at::Tensor& self, const int64_t dim, const at::Tensor& index, const at::Tensor& source)
{
    at::Tensor contiguous_self(self.clone());
    if (!npu_utils::check_match(&self)) {
        contiguous_self = npu_utils::format_contiguous(contiguous_self);
    }
    return index_copy_npu_impl(dim, index, source, contiguous_self);
}
#endif

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor &_index_copy_(at::Tensor &self, const int64_t dim, const at::Tensor &index, const at::Tensor &source)
{
    at::Tensor contiguous_self(self);
    if (!npu_utils::check_match(&self)) {
        contiguous_self = npu_utils::format_contiguous(self);
    }
    at::Tensor result = index_copy_npu_impl(dim, index, source, contiguous_self);
    npu_utils::format_fresh_view(self, result);

    return self;
}
#endif

#if VERSION_BETWEEN(V2R0, V2R0)
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
#endif

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
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
#endif
} // namespace acl_op
