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
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& any_out(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnAny, acl_op::any_out(self, dim, keepdim, out));
    c10::SmallVector<int64_t, op_infer::N> dim_list = {dim};

    // check result for return
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keepdim);
    npu_preparation::check_tensor({self}, out, out, output_size);

    // calculate the output result of the NPU
    at::IntArrayRef dims(dim);
    EXEC_NPU_CMD(aclnnAny, self, dims, keepdim, out);
    return out;
}

at::Tensor& any_out(const at::Tensor& self, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnAny, acl_op::any_out(self, out));
    at::SmallVector<int64_t, op_infer::N> dim_list = op_plugin::utils::get_dimlist_for_tensor(self);
    bool keep_dim = false;

    // check result for return
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keep_dim);
    npu_preparation::check_tensor({self}, out, out, output_size);
    at::IntArrayRef dims(dim_list);
    EXEC_NPU_CMD(aclnnAny, self, dims, keep_dim, out);
    return out;
}

at::Tensor any(const at::Tensor& self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnAny, acl_op::any(self, dim, keepdim));

    // calculate the output size
    at::IntArrayRef dims(dim);
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dims, keepdim);
    auto output_dtype = (self.scalar_type() == at::ScalarType::Byte) ? at::ScalarType::Byte : at::ScalarType::Bool;
    auto options = self.options().dtype(output_dtype);

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);
    EXEC_NPU_CMD(aclnnAny, self, dims, keepdim, result);
    return result;
}

at::Tensor any(const at::Tensor& self)
{
    DO_COMPATIBILITY(aclnnAny, acl_op::any(self));
    at::SmallVector<int64_t, op_infer::N> dim_list = op_plugin::utils::get_dimlist_for_tensor(self);
    bool keep_dim = false;
    auto output_size = op_infer::reduce_ops_npu_output_size(self, dim_list, keep_dim);
    auto output_dtype = (self.scalar_type() == at::ScalarType::Byte) ? at::ScalarType::Byte : at::ScalarType::Bool;
    auto options = self.options().dtype(output_dtype);

    // construct the output tensor of the NPU
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, options);
    at::IntArrayRef dims(dim_list);
    EXEC_NPU_CMD(aclnnAny, self, dims, keep_dim, result);
    return result;
}
}
