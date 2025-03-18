// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
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
using npu_preparation = at_npu::native::OpPreparation;

std::vector<at::Tensor> _foreach_addcdiv(const at::TensorList input,
                                         const at::TensorList tensors1,
                                         const at::TensorList tensors2,
                                         at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars);
    return at::native::foreach_tensor_addcdiv_scalarlist_slow(input, tensors1, tensors2, scalars);
}

void _foreach_addcdiv_(const at::TensorList input,
                       const at::TensorList tensors1,
                       const at::TensorList tensors2,
                       at::ArrayRef<at::Scalar> scalars)
{
    at::native::check_foreach_api_restrictions(input, tensors1, tensors2, scalars);
    return at::native::foreach_tensor_addcdiv_scalarlist_slow_(input, tensors1, tensors2, scalars);
}
}
