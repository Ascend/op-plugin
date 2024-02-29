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

#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::ScalarType angle_out_dtype(const at::Tensor& self)
{
    auto out_dtype = self.scalar_type();
    if (self.is_complex()) {
        out_dtype = self.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    } else if (at::isIntegralType(out_dtype, true)) {
        out_dtype = at::kFloat;
    }
    return out_dtype;
}

at::Tensor& angle_out(const at::Tensor& self, at::Tensor& result)
{
    auto output_size = self.sizes();
    auto out_check_dtype = angle_out_dtype(self);
    npu_preparation::check_tensor({self}, result, out_check_dtype, output_size);
    EXEC_NPU_CMD(aclnnAngleV2, self, result);
    return result;
}

at::Tensor angle(const at::Tensor& self)
{
    auto output_size = self.sizes();
    auto out_dtype = angle_out_dtype(self);
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, self.options().dtype(out_dtype));
    EXEC_NPU_CMD(aclnnAngleV2, self, result);
    return result;
}

} // namespace op_api
