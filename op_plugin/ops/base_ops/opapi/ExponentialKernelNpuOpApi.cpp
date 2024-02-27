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
#include "torch_npu/csrc/framework/utils/RandomOpAdapter.h"

namespace op_api {
at::Tensor& exponential_(at::Tensor& self, double lambda, c10::optional<at::Generator> generator)
{
    self = op_api::uniform_(self, 0.0, 1.0, generator);
    self = op_api::sub_(self, at::Scalar(1.0), at::Scalar(1.0));
    self = op_api::mul_(self, at::Scalar(-1.0));
    self = op_api::log_(self);
    return op_api::div_(self, at::Scalar(-lambda));
}
}  // namespace op_api
