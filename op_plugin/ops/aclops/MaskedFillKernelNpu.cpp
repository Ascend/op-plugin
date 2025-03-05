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
using npu_utils = at_npu::native::NpuUtils;
using npu_preparation = at_npu::native::OpPreparation;

namespace {
at::Tensor& masked_fill_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    at::Tensor mask_bool = mask;
    int64_t dim_of_self = self.dim();
    /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
    if (dim_of_self == 0) {
        self.unsqueeze_(0);
    }

    if ((mask.dtype() != at::kBool)) {
        mask_bool = at_npu::native::custom_ops::npu_dtype_cast(mask, at::kBool);
    }
    at::Tensor value_tensor = value;
    if (value.dtype() != self.dtype()) {
        value_tensor = value_tensor.to(self.dtype());
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("MaskedFill")
        .Input(self)
        .Input(mask_bool)
        .Input(value_tensor)
        .Output(result)
        .Run();

    if (dim_of_self == 0) {
        result.squeeze_(0);
    }
    return result;
}

at::Tensor& masked_fill_out_npu_nocheck(at::Tensor& result, const at::Tensor& self, const at::Tensor& mask, at::Scalar value)
{
    at::Tensor mask_bool = mask;
    int64_t dim_of_self = self.dim();
    /* Avoid the problem that the TBE operator does not support 0-dimensional tensor input */
    if (dim_of_self == 0) {
        self.unsqueeze_(0);
    }

    if (!(mask.dtype() == at::kBool)) {
        mask_bool = at_npu::native::custom_ops::npu_dtype_cast(mask, at::kBool);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("MaskedFill")
        .Input(self)
        .Input(mask_bool)
        .Input(value, self.scalar_type())
        .Output(result)
        .Run();

    if (dim_of_self == 0) {
        result.squeeze_(0);
    }
    return result;
}
} // namespace

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value)
{
    if (npu_preparation::IsCPUScalar(value)) {
        return acl_op::masked_fill_(self, mask, value.item());
    }

    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        masked_fill_out_npu_nocheck(contiguous_self, contiguous_self, mask, value);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        masked_fill_out_npu_nocheck(self, self, mask, value);
    }
    return self;
}

at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value)
{
    if (!npu_utils::check_match(&self)) {
        at::Tensor contiguous_self = npu_utils::format_contiguous(self);
        masked_fill_out_npu_nocheck(contiguous_self, contiguous_self, mask, value);
        npu_utils::format_fresh_view(self, contiguous_self);
    } else {
        masked_fill_out_npu_nocheck(self, self, mask, value);
    }

    return self;
}
} // namespace acl_op
