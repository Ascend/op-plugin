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
#include "op_plugin/utils/OpAdapter.h"

namespace op_api {

at::Tensor softmax(const at::Tensor& self, int64_t dim, c10::optional<at::ScalarType> dtype)
{
    DO_COMPATIBILITY(aclnnSoftmax, acl_op::softmax(self, dim, dtype));
    auto result = [&]() {
        at::NoNamesGuard guard;
        at::Tensor converted = dtype.has_value() ? at_npu::native::custom_ops::npu_dtype_cast(self, dtype.value()) : self;
        return at::_softmax(converted, dim, false);
    }();
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor softmax(const at::Tensor& self, at::Dimname dim, c10::optional<at::ScalarType> dtype)
{
    return op_api::softmax(self, dimname_to_position(self, dim), dtype);
}

}
