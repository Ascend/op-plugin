// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <c10/util/irange.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_gather_backward_symint(const at::Tensor& grad, c10::SymIntArrayRef self_size, int64_t dim, const at::Tensor& index, bool sparse_grad)
{
    if (sparse_grad) {
        const int64_t self_dim = self_size.size();
        if (self_dim == 0) {
            return at::_sparse_coo_tensor_unsafe_symint(
                at::empty_symint({0, grad.sym_numel()}, index.options()),
                grad,
                self_size);
        }
        if (grad.ndimension() == 0) {
            return at::_sparse_coo_tensor_unsafe_symint(
                index.view({1, 1}),
                grad,
                self_size);
        }

        at::Tensor sparse_ind = at::empty_symint(
            {self_dim, grad.sym_numel()},
            grad.options().dtype(at::kLong));
        c10::SymInt grad_numel = grad.sym_numel();
        if (grad_numel > 0) {
            c10::SymInt n_above = grad_numel;
            c10::SymInt n_below = 1;
            if (dim < 0) {
                dim += self_dim;
            }
            for (const auto i : c10::irange(self_dim)) {
                n_above /= grad.sym_size(i);
                if (i == dim) {
                    sparse_ind[i] = index.reshape(-1);
                } else {
                    sparse_ind[i] =
                        at::arange(grad.sym_size(i), grad.options().dtype(at::kLong))
                            .unsqueeze(1)
                            .expand_symint({grad.sym_size(i), n_above})
                            .reshape(-1)
                            .repeat_symint(n_below);
                }
                n_below *= grad.sym_size(i);
            }
        }
        return at::_sparse_coo_tensor_unsafe_symint(
            sparse_ind,
            grad.reshape(-1),
            self_size);
    }

    auto result = grad.new_zeros_symint(self_size);
    // for composite, vmap and inductor compliance, use out-of-place variant of
    // `scatter_add` if index or grad tensors is a Tensor Subclass.
    if (at::areAnyTensorSubclassLike({index, grad})) {
        return result.scatter_add(dim, index, grad);
    }
    result.scatter_add_(dim, index, grad);
    return result;
}
}
