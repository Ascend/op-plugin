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

at::Tensor count_nonzero(
    const at::Tensor& self,
    c10::IntArrayRef dim)
{
    return op_api::sum((self != 0), dim, false, at::ScalarType::Long);
}

at::Tensor count_nonzero(
    const at::Tensor& self,
    c10::optional<int64_t> dim)
{
    if (dim.has_value()) {
        return op_api::count_nonzero(self, at::IntArrayRef{dim.value()});
    }
    return op_api::count_nonzero(self, at::IntArrayRef{});
}

}
