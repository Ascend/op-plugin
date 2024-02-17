// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor& nansum_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                       c10::optional<c10::ScalarType> dtype, at::Tensor& result)
{
    at::IntArrayRef dimArray;
    c10::SmallVector<int64_t, N> dimlist;
    if (dim.has_value()) {
        dimArray = dim.value();
    } else {
        dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
        dimArray = dimlist;
    }

    c10::ScalarType dstType;
    if (dtype.has_value()) {
        dstType = dtype.value();
    } else if (result.defined()) {
        dstType = result.scalar_type();
    } else {
        dstType = self.scalar_type();
    }
    // infer reduecshape
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), output_size);

    EXEC_NPU_CMD(aclnnReduceNansum, self, dimArray, keepdim, dstType, result);
    return result;
}

at::Tensor nansum(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                  c10::optional<c10::ScalarType> dtype)
{
    // create result tensor with int64 if dtype has no value and self is inte
    c10::ScalarType promoteInteSelf = isIntegralType(self.scalar_type(), true) ? at::kLong : self.scalar_type();
    c10::ScalarType dstType = dtype.has_value() ? dtype.value() : promoteInteSelf;
    at::IntArrayRef dimArray;
    c10::SmallVector<int64_t, N> dimlist;
    if (dim.has_value()) {
        dimArray = dim.value();
    } else {
        dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
        dimArray = dimlist;
    }
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(output_size, self.options().dtype(dstType));

    // calculate the output result of the NPU
    op_api::nansum_out(self, dim, keepdim, dtype, result);
    return result;
}

}  // namespace op_api
