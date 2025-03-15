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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

#if VERSION_BETWEEN(V1R11, V1R11)
at::Tensor& mean_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype,
                     at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean_out(self, dim, keepdim, dtype, result));
    c10::ScalarType dstType;
    if (dtype.has_value()) {
        dstType = dtype.value();
    } else if (result.defined()) {
        dstType = result.scalar_type();
    } else {
        dstType = self.scalar_type();
    }
    // �Ƶ�reduecshape
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, result, result.scalar_type(), outputSize);

    EXEC_NPU_CMD(aclnnMean, self, dim, keepdim, dstType, result);
    return result;
}

at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dim, keepdim, dtype));
    c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();

    // calculate the output size
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dim, keepdim);

    // construct the output tensor of the NPU
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(outputSize, self.options().dtype(dstType));

    // calculate the output result of the NPU
    op_api::mean_out(self, dim, keepdim, dtype, result);
    return result;
}

at::Tensor mean(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dim, keepdim, dtype));
    return op_api::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor& mean_out(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype,
                     at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean_out(self, dim, keepdim, dtype, result));
    return op_api::mean_out(self, dimnames_to_positions(self, dim), keepdim, dtype, result);
}

at::Tensor mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dtype));
    return op_api::mean(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}
#endif

#if VERSION_BETWEEN(V2R0, VERSION_NEWEST)
at::Tensor& mean_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                     c10::optional<c10::ScalarType> dtype, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean_out(self, dim, keepdim, dtype, out));
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
    } else if (out.defined()) {
        dstType = out.scalar_type();
    } else {
        dstType = self.scalar_type();
    }
    // reduecshape
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);
    at_npu::native::OpPreparation::check_tensor({self}, out, out.scalar_type(), outputSize);

    EXEC_NPU_CMD(aclnnMean, self, dimArray, keepdim, dstType, out);
    return out;
}

at::Tensor mean(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dim, keepdim, dtype));
    c10::ScalarType dstType = dtype.has_value() ? dtype.value() : self.scalar_type();
    at::IntArrayRef dimArray;
    c10::SmallVector<int64_t, N> dimlist;
    if (dim.has_value()) {
        dimArray = dim.value();
    } else {
        dimlist = op_plugin::utils::get_dimlist_for_tensor(self);
        dimArray = dimlist;
    }
    // calculate the output size
    auto outputSize = op_infer::reduce_ops_npu_output_size(self, dimArray, keepdim);

    // construct the output tensor of the NPU
    at::Tensor result =
        at_npu::native::OpPreparation::apply_tensor_without_format(outputSize, self.options().dtype(dstType));

    // calculate the output result of the NPU
    op_api::mean_out(self, dim, keepdim, dtype, result);
    return result;
}

at::Tensor mean(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dim, keepdim, dtype));
    return op_api::mean(self, dimnames_to_positions(self, dim), keepdim, dtype);
}

at::Tensor& mean_out(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype,
                     at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean_out(self, dim, keepdim, dtype, out));
    return op_api::mean_out(self, dimnames_to_positions(self, dim), keepdim, dtype, out);
}

at::Tensor mean(const at::Tensor& self, c10::optional<c10::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnMean, acl_op::mean(self, dtype));
    return op_api::mean(self, c10::SmallVector<int64_t, N>{}, false, dtype);
}
#endif

}
