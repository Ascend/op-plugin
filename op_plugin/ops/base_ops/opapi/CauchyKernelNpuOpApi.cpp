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

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
 
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
#define PI 3.141592653
 
at::Tensor& cauchy_(at::Tensor& self, double median, double sigma, c10::optional<at::Generator> gen)
{
    self = op_api::uniform_(self, 0.0, 1.0, gen);
    self = op_api::sub_(self, at::Scalar(0.5), at::Scalar(1.0));
    self = op_api::tan_(op_api::mul_(self, at::Scalar(PI)));
    return op_api::add_(op_api::mul_(self, at::Scalar(sigma)), at::Scalar(median), at::Scalar(1.0));
}
}  // namespace op_api
