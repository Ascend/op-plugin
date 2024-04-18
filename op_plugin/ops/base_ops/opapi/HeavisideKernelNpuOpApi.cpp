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
 
at::Tensor& heaviside_out(const at::Tensor& input, const at::Tensor& value, at::Tensor& result)
{
    at::Tensor input_01 = acl_op::eq(input, at::Scalar(0));
    result = acl_op::mul_out(input_01, value, result);
    return op_api::add_(result, acl_op::gt(input, at::Scalar(0)), at::Scalar(1.0));
}
 
at::Tensor heaviside(const at::Tensor& input, const at::Tensor& value)
{
    at::Tensor result = npu_preparation::apply_tensor(value);
    op_api::heaviside_out(input, value, result);
    return result;
}
}  // namespace op_api
