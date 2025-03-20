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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace {

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
// Sorting-based algorithm for isin(); used when the number of test elements is large.
static void isin_sorting(const at::Tensor& elements,
                         const at::Tensor& test_elements,
                         bool assume_unique,
                         bool invert,
                         const at::Tensor& out)
{
    // 1. Concatenate unique elements with unique test elements in 1D form. If
    //    assume_unique is true, skip calls to unique().
    at::Tensor elements_flat;
    at::Tensor test_elements_flat;
    at::Tensor unique_order;
    if (assume_unique) {
        elements_flat = elements.ravel();
        test_elements_flat = test_elements.ravel();
    } else {
        std::tie (elements_flat, unique_order) = at::_unique(elements, false, true);
        std::tie (test_elements_flat, std::ignore) = at::_unique(test_elements, false);
    }

    // 2. Stable sort all elements, maintaining order indices to reverse the
    //    operation. Stable sort is necessary to keep elements before test
    //    elements within the sorted list.
    at::Tensor all_elements = at::cat({std::move(elements_flat), std::move(test_elements_flat)});
    at::Tensor sorted_elements;
    at::Tensor sorted_order;
    std::tie (sorted_elements, sorted_order) = all_elements.sort(true, 0, false);

    // 3. Create a mask for locations of adjacent duplicate values within the
    //    sorted list. Duplicate values are in both elements and test elements.
    at::Tensor duplicate_mask = at::empty_like(sorted_elements, at::TensorOptions(at::ScalarType::Bool));
    at::Tensor sorted_except_first = sorted_elements.slice(0, 1, at::indexing::None);
    at::Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
    duplicate_mask.slice(0, 0, -1).copy_(
        invert ? sorted_except_first.ne(sorted_except_last) : sorted_except_first.eq(sorted_except_last));
    duplicate_mask.index_put_({-1}, invert);

    // 4. Reorder the mask to match the pre-sorted element order.
    at::Tensor mask = at::empty_like(duplicate_mask);
    mask.index_copy_(0, sorted_order, duplicate_mask);

    // 5. Index the mask to match the pre-unique element order. If
    //    assume_unique is true, just take the first N items of the mask,
    //    where N is the original number of elements.
    if (assume_unique) {
        out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
    } else {
        out.copy_(at::index(mask, {c10::optional<at::Tensor>(unique_order)}));
    }
}

void isin_default_kernel_npu(const at::Tensor& elements,
                             const at::Tensor& test_elements,
                             bool invert,
                             const at::Tensor& out)
{
    std::vector<int64_t> bc_shape(elements.dim(), 1);
    bc_shape.push_back(-1);
    out.copy_(invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
                : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

void isin_Tensor_Tensor_out_impl(const at::Tensor& elements,
                                 const at::Tensor& test_elements,
                                 bool assume_unique,
                                 bool invert,
                                 const at::Tensor& out)
{
    if (elements.numel() == 0) {
        return;
    }

    // Heuristic taken from numpy's implementation.
    // See https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
    if (test_elements.numel() < static_cast<int64_t>(
            10.0f * std::pow(static_cast<double>(elements.numel()), 0.145))) {
        out.fill_(invert);
        isin_default_kernel_npu(elements, test_elements, invert, out);
    } else {
        isin_sorting(elements, test_elements, assume_unique, invert, out);
    }
}
#endif
} // namespace

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

#if VERSION_BETWEEN(V2R1, VERSION_NEWEST)
at::Tensor& isin_out(const at::Tensor& elements, const at::Tensor &test_elements,
                     bool assume_unique, bool invert, at::Tensor& result)
{
    isin_Tensor_Tensor_out_impl(elements, test_elements, assume_unique, invert, result);
    return result;
}
at::Tensor isin(const at::Tensor& elements, const at::Tensor &test_elements,
                bool assume_unique, bool invert)
{
    at::Tensor result = npu_preparation::apply_tensor_without_format(
        elements.sizes(),
        elements.options().dtype(at::kBool));
    isin_Tensor_Tensor_out_impl(elements, test_elements, assume_unique, invert, result);
    return result;
}
#endif
}
