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

#include "torch_npu/csrc/aten/CustomFunctions.h"

#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

at::Tensor dropout(const at::Tensor& self, double p, bool train)
{
    if (p == 0 || !train || self.numel() == 0) {
        return self;
    }
    if (p == 1) {
        return self.mul(at::zeros(self.sizes(), self.options()));
    }
    auto results = at_npu::native::custom_ops::_npu_dropout(self, p);
    return std::get<0>(results);
}

at::Tensor& dropout_(at::Tensor& self, double p, bool train)
{
    if (p == 0 || !train || self.numel() == 0) {
        return self;
    }
    if (p == 1) {
        return self.mul_(at::zeros(self.sizes(), self.options()));
    }
    auto results = at_npu::native::custom_ops::_npu_dropout(self, p);
    self.copy_(std::get<0>(results));
    return self;
}
}  // namespace op_api
