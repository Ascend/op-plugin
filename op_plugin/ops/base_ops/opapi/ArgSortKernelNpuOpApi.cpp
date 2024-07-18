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

at::Tensor argsort(const at::Tensor &self, int64_t dim, bool descending)
{
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        TORCH_NPU_WARN_ONCE("Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel "
                            "is running on AiCpu."
                            "If you are more concerned about high-performance execution,please cast dtype to float32.");
    }
    return std::get<1>(at::sort(self, dim, descending));
}

at::Tensor argsort(const at::Tensor &self, at::Dimname dim, bool descending)
{
    return op_api::argsort(self, dimname_to_position(self, dim), descending);
}

at::Tensor argsort(const at::Tensor &self, bool stable, int64_t dim, bool descending)
{
    if (self.dtype() == at::kInt || self.dtype() == at::kLong) {
        TORCH_NPU_WARN_ONCE("Warning: kernel [ArgSort] can not support dtype int32 or int64 on AiCore, Now this kernel "
                            "is running on AiCpu."
                            "If you are more concerned about high-performance execution,please cast dtype to float32.");
    }
    return std::get<1>(at::sort(self, stable, dim, descending));
}
} // namespace op_api
