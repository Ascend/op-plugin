// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {

std::tuple<at::Tensor, at::Tensor> native_dropout(
    const at::Tensor& input,
    double p,
    c10::optional<bool> train)
{
    if (input.numel() == 0) {
        return std::make_tuple(input, at::empty_like(input, input.options()));
    }

    bool dropout_train = !train.has_value() ? true : train.value();

    at::TensorOptions options = input.options();
    if (p == static_cast<double>(0) || !dropout_train) {
        at::Tensor mask = acl_op::ones(
            input.sizes(),
            at::kBool,
            options.layout(),
            options.device(),
            options.pinned_memory());
        return std::make_tuple(input.clone(), mask);
    }
    if (p == static_cast<double>(1)) {
        at::Tensor output = at::zeros(input.sizes(), options);
        at::Tensor mask = at::zeros(input.sizes(), options.dtype(at::kBool));
        return std::make_tuple(output, mask);
    }

    return acl_op::_npu_dropout(input, p);
}

at::Tensor native_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale)
{
    double p = (scale == static_cast<double>(0.0)) ? 1 : (1 - 1 / scale);
    at::TensorOptions options = grad_output.options();
    if (p == static_cast<double>(0)) {
        return grad_output;
    }
    if (p == static_cast<double>(1)) {
        return at::zeros(grad_output.sizes(), options);
    }
    return acl_op::npu_dropout_backward(grad_output, mask, p);
}

} // namespace acl_op
