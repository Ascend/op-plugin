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
using npu_utils = at_npu::native::NpuUtils;
using npu_compile_type = at_npu::native::CompileType;

namespace {
// format are base format (the format of src and dst are all nchw now)
// dtype are same
// so the view_value and ReflushDescBySelf are base on the hypothesis above.
bool AicoreValid(at::Tensor &self, const at::Tensor &src)
{
    const auto &dst_storage_sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    auto self_size = self.sizes();
    auto self_stride = self.strides();
    auto dst_storage_size_len = dst_storage_sizes.size();
    auto self_size_len = self_size.size();

    // count the difference between dst_storage and dst_size.
    auto diff = dst_storage_size_len - self_size_len;
    if (diff < 0 || diff > 1) {
        return false;
    }

    // record the index of the difference.
    auto diff_index = self_size_len;
    for (uint64_t i = 0; i < self_size_len; i++) {
        if (dst_storage_sizes[i] != self_size[i]) {
            ++diff;
            if (diff > 1) {
                return false;
            }
            // differece should be 1.
            diff_index = i;
        }
    }

    // if diff or diff_index equals 0, no need viewcopy.
    if (diff == 0 || diff_index == 0) {
        return false;
    }

    const auto &dst_base_stride = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.base_strides_;
    // dst_base_stride should be equal to dst_storage_stride except for diff_index
    if (self_stride.size() > dst_base_stride.size()) {
        return false;
    }

    for (uint64_t i = 0; i < self_stride.size(); i++) {
        if (dst_base_stride[i] != self_stride[i] && i != diff_index) {
            return false;
        }
    }

    // dtype cannot be double and dst_size has to be equal with src_size.
    if (self.dtype() == at::ScalarType::Double || self_size != src.sizes()) {
        return false;
    }

    return true;
}
} // namespace

at::Tensor &npu_view_copy(at::Tensor &self, const at::Tensor &other, bool non_blocking)
{
    auto self_size = self.sizes();
    auto self_stride = self.strides();
    auto src_size = other.sizes();
    auto src_stride = other.strides();

    at_npu::native::OpCommand cmd;
    if (AicoreValid(self, other)) {
        at::Tensor contiguous_src(other);
        if (!npu_utils::check_match(&contiguous_src)) {
            contiguous_src = npu_utils::format_contiguous(contiguous_src);
        }
        src_stride = contiguous_src.strides();

        cmd.Name("ViewCopy")
            .InputWithoutContiguous(self)
            .Input(self_size, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(self_stride, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(at::Scalar(0), at::kLong)
            .InputWithoutContiguous(contiguous_src)
            .Input(src_size, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(src_stride, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(at::Scalar(0), at::kLong)
            .Output(self)
            .Run();
    } else {
        cmd.Name("ViewCopy")
            .InputWithoutContiguous(self)
            .Input(self_size, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(self_stride, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(at::Scalar(0), at::kLong)
            .InputWithoutContiguous(other)
            .Input(src_size, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(src_stride, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_INDEPENDENT)
            .Input(at::Scalar(0), at::kLong)
            .Output(self)
            .Attr("_exclude_engines", static_cast<string>("AiCore"))
            .Run();
    }

    return self;
}
} // namespace acl_op
