// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

static at::DimVector default_alldims(const at::Tensor& self, at::OptionalIntArrayRef dim_opt)
{
    at::DimVector dim;
    if (dim_opt.has_value()) {
        at::IntArrayRef dim_unwrapped = dim_opt.value();
        dim.resize(dim_unwrapped.size());
        for (const auto i : c10::irange(dim.size())) {
            dim[i] = at::maybe_wrap_dim(dim_unwrapped[i], self.dim(), false);
        }
    } else {
        dim.resize(self.dim());
        std::iota(dim.begin(), dim.end(), 0);
    }
    return dim;
}

at::Tensor fft_fftshift(const at::Tensor& x, at::OptionalIntArrayRef dim_opt)
{
    auto dim = default_alldims(x, dim_opt);

    at::SymIntArrayRef x_sizes = x.sym_sizes();
    at::SymDimVector shift(dim.size());
    for (const auto i : c10::irange(dim.size())) {
        shift[i] = x_sizes[dim[i]] / 2;
    }

    if (x.scalar_type() == at::ScalarType::ComplexFloat) {
        auto res = x.view(x.sizes());
        auto *impl = res.unsafeGetTensorImpl();
        impl->set_storage_and_dtype(res.storage(), c10::scalarTypeToTypeMeta(at::ScalarType::Long));
        res = at::roll_symint(res, shift, dim);
        impl = res.unsafeGetTensorImpl();
        impl->set_storage_and_dtype(res.storage(), c10::scalarTypeToTypeMeta(at::ScalarType::ComplexFloat));
        return res;
    }

    return at::roll_symint(x, shift, dim);
}

at::Tensor fft_ifftshift(const at::Tensor& x, at::OptionalIntArrayRef dim_opt)
{
    auto dim = default_alldims(x, dim_opt);

    at::SymIntArrayRef x_sizes = x.sym_sizes();
    at::SymDimVector shift(dim.size());
    for (const auto i : c10::irange(dim.size())) {
        shift[i] = (x_sizes[dim[i]] + 1) / 2;
    }

    if (x.scalar_type() == at::ScalarType::ComplexFloat) {
        auto res = x.view(x.sizes());
        auto *impl = res.unsafeGetTensorImpl();
        impl->set_storage_and_dtype(res.storage(), c10::scalarTypeToTypeMeta(at::ScalarType::Long));
        res = at::roll_symint(res, shift, dim);
        impl = res.unsafeGetTensorImpl();
        impl->set_storage_and_dtype(res.storage(), c10::scalarTypeToTypeMeta(at::ScalarType::ComplexFloat));
        return res;
    }

    return at::roll_symint(x, shift, dim);
}

}  // namespace op_api