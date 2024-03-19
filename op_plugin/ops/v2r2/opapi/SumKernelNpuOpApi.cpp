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
#include "op_plugin/utils/custom_functions/opapi/inner_compute_op_api.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor& sum_out(const at::Tensor &self,
                    at::OptionalIntArrayRef dim,
                    bool keepdim,
                    c10::optional<c10::ScalarType> dtype,
                    at::Tensor &result) {
    return op_api::sum_out_common_nocheck(self, dim.value_or(at::IntArrayRef{}), keepdim, dtype, result);
}

at::Tensor sum(const at::Tensor &self,
               at::OptionalIntArrayRef dim,
               bool keepdim,
               c10::optional<c10::ScalarType> dtype) {
    return op_api::sum_common_nocheck(self, dim.value_or(at::IntArrayRef{}), keepdim, dtype);
}
} // namespace op_api
