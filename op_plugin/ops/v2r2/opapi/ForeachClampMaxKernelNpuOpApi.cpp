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

#include <ATen/native/ForeachUtils.h>
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {

std::vector<at::Tensor> _foreach_clamp_max(at::TensorList tensors1, at::TensorList tensors2)
{
    return op_api::_foreach_minimum(tensors1, tensors2);
}

void _foreach_clamp_max_(at::TensorList tensors1, at::TensorList tensors2)
{
    op_api::_foreach_minimum_(tensors1, tensors2);
    return;
}

std::vector<at::Tensor> _foreach_clamp_max(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    return op_api::_foreach_minimum(tensors, scalars);
}

void _foreach_clamp_max_(at::TensorList tensors, at::ArrayRef<at::Scalar> scalars)
{
    op_api::_foreach_minimum_(tensors, scalars);
    return;
}
}  // namespace op_api
