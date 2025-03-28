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
#include "op_plugin/utils/custom_functions/aclops/inner_compute.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;

std::tuple<at::Tensor&, at::Tensor&> min_v1_out_npu_nocheck(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t dim,
    bool keepdim)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMinWithValue")
        .Input(self)
        .Output(indices)
        .Output(output)
        .Attr("dimension", dim)
        .Attr("keep_dims", keepdim)
        .Run();
    return std::tie(output, indices);
}

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor npu_min_backward(
    const at::Tensor& grad,
    int64_t dim,
    const at::Tensor& indices,
    at::IntArrayRef sizes,
    bool keepdim)
{
    at::Tensor new_grad = grad;
    at::Tensor new_indices = indices;
    if (keepdim && sizes.size() > 0) {
        new_grad = grad.squeeze(dim);
        new_indices = indices.squeeze(dim);
    }
    auto grad_input = acl_op::npu_scatter(
        at::zeros(sizes, new_grad.options()), new_indices, new_grad, dim);
    return grad_input;
}
#endif

std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, at::Dimname dim, bool keepdim)
{
    return acl_op::npu_min(self, dimname_to_position(self, dim), keepdim);
}

std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, int64_t dim, bool keepdim)
{
    c10::SmallVector<int64_t, SIZE> dims = {dim};
    c10::SmallVector<int64_t, SIZE> output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    c10::SmallVector<int64_t, SIZE> indices_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);

    int64_t npu_format = output_size.empty() ? ACL_FORMAT_NCHW : npu_preparation::get_tensor_npu_format(self);

    at::Tensor outputs = npu_preparation::apply_tensor_with_format(output_size, self.options(), npu_format);
    at::Tensor indices =
        npu_preparation::apply_tensor_with_format(indices_size, self.options().dtype(at::kInt), npu_format);

    min_v1_out_npu_nocheck(outputs, indices, self, dim, keepdim);
    return std::tie(outputs, indices);
}

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor npu_min_backward_symint(const at::Tensor &grad, int64_t dim, const at::Tensor &indices,
                                   c10::SymIntArrayRef sizes, bool keepdim)
{
    at::IntArrayRef sizes_ = c10::asIntArrayRefUnchecked(sizes);
    at::Tensor new_grad = grad;
    at::Tensor new_indices = indices;
    if (keepdim && sizes_.size() > 0) {
        new_grad = grad.squeeze(dim);
        new_indices = indices.squeeze(dim);
    }
    auto grad_input = acl_op::npu_scatter(at::zeros(sizes_, new_grad.options()), new_indices, new_grad, dim);
    return grad_input;
}
#endif
} // namespace acl_op
