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

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
    using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_quant_scatter(
    const at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    const at::Tensor& quant_scales,
    const c10::optional<at::Tensor>& quant_zero_points,
    int64_t axis,
    int64_t quant_axis,
    c10::string_view reduce)
{
    at::Tensor result = self.clone();
    int64_t reduction = 1;
    EXEC_NPU_CMD(aclnnInplaceQuantScatter, result, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                 reduction);
    return result;
}

at::Tensor& npu_quant_scatter_(
    at::Tensor& self,
    const at::Tensor& indices,
    const at::Tensor& updates,
    const at::Tensor& quant_scales,
    const c10::optional<at::Tensor>& quant_zero_points,
    int64_t axis,
    int64_t quant_axis,
    c10::string_view reduce)
{
    int64_t reduction = 1;
    EXEC_NPU_CMD(aclnnInplaceQuantScatter, self, indices, updates, quant_scales, quant_zero_points, axis, quant_axis,
                 reduction);
    return self;
}

}
