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

at::Tensor& nanmean_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                        c10::optional<c10::ScalarType> dtype, at::Tensor& out)
{
    // Check if dtype is an integral type or Bool and raise an error
    TORCH_CHECK(
        !at::isIntegralType(self.scalar_type(), /*includeBool=*/true),
        "nanmean(): integral types and 'Bool' are not supported for nanmean, even for empty tensors.");
    TORCH_CHECK(
        self.is_floating_point() || self.is_complex(),
        "nanmean(): expected input to have floating point or complex dtype but got ",
        self.scalar_type());

    // Calculate the factor: count of non-NaN values
    const auto factor = at::isnan(self).logical_not_().sum(dim, keepdim);

    // Compute nansum and divide by factor in-place
    op_api::nansum_out(self, dim, keepdim, dtype, out).div_(factor);

    return out;
}
}