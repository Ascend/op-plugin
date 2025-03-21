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

#include "torch_npu/csrc/framework/utils/InternalFormatOpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_compile_type = at_npu::native::CompileType;

namespace {
at::Tensor& stride_copy_out_npu_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    at::IntArrayRef shape,
    at::IntArrayRef stride,
    at::Scalar storage_offset)
{
    if ((result.nbytes() < 32) && (!at_npu::native::StorageDescHelper::MetaDataAreMatch(&result))) {
        // [算子限制] 对于1. 小于一个block的数据搬运 2.result不match，Astrided暂不支持。
        acl_op::npu_view_copy(result, self, false);
        return result;
    }

    at_npu::native::OpCommand cmd;
    // When the last dimension of the input tensor stride is greater than 256, we use
    // AsStrided + Transpose instead of AsStrided to get better performance.
    if (stride.back() >= 256) {
        std::set<std::pair<int, std::pair<int, int>>> stride_perm_shape_set;
        int tensor_dim = static_cast<int>(stride.size());
        for (int i = 0; i < tensor_dim; i++) {
        stride_perm_shape_set.insert({stride[i], {i, shape[i]}});
        }
        std::set<std::pair<int, std::pair<int, int>>>::reverse_iterator iter;
        std::vector<int64_t> output_stride;
        std::vector<int64_t> output_shape;
        std::vector<int64_t> output_perm_origin;
        for (iter = stride_perm_shape_set.rbegin(); iter != stride_perm_shape_set.rend(); iter++) {
        output_stride.emplace_back((*iter).first);
        output_shape.emplace_back((*iter).second.second);
        output_perm_origin.emplace_back((*iter).second.first);
        }
        at::IntArrayRef output_stride_array(output_stride);
        at::IntArrayRef output_shape_array(output_shape);
        at::Tensor result_out = npu_preparation::apply_tensor_with_format(
            output_shape_array, self.options(), ACL_FORMAT_ND);
        at_npu::native::NpuStorageOffsetGuard guard_input(const_cast<at::Tensor &>(self));
        cmd.Name("AsStrided")
        .InputWithoutContiguous(self)
        .Input(output_shape_array)
        .Input(output_stride_array)
        .Input(storage_offset, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result_out)
        .Run();
        std::vector<int64_t> output_perm(tensor_dim);
        for (auto i = 0; i < tensor_dim; i++) {
        output_perm[output_perm_origin[i]] = i;
        }
        at::IntArrayRef output_perm_array(output_perm);
        result = acl_op::npu_transpose(result_out, output_perm_array, true);
        return result;
    } else {
        at_npu::native::NpuStorageOffsetGuard guard_input(const_cast<at::Tensor &>(self));
        cmd.Name("AsStrided")
        .InputWithoutContiguous(self)
        .Input(shape)
        .Input(stride)
        .Input(storage_offset, at::kLong, npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Run();
        return result;
    }
}
} // namespace

at::Tensor& npu_stride_copy_out(
    const at::Tensor& self,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    const c10::Scalar& storage_offset,
    at::Tensor& out)
{
    stride_copy_out_npu_nocheck(out, self, shape, stride, storage_offset);
    return out;
}

at::Tensor npu_stride_copy(
    const at::Tensor& self,
    c10::IntArrayRef shape,
    c10::IntArrayRef stride,
    const c10::Scalar& storage_offset)
{
    // AsStrided OP only supports ND input
    at::Tensor result = at_npu::native::OpPreparation::apply_tensor_with_format(
        shape, self.options(), ACL_FORMAT_ND);
    stride_copy_out_npu_nocheck(result, self, shape, stride, storage_offset);
    return result;
}

} // namespace acl_op
