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
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& softmax_out_nocheck(
    at::Tensor& result,
    const at::Tensor& self,
    int64_t dim)
{
    c10::SmallVector<int64_t, N> dim_list = {dim};
    at_npu::native::OpCommand cmd;
    cmd.Name("SoftmaxV2")
        .Input(self)
        .Output(result)
        .Attr("axes", dim_list)
        .Run();
    return result;
}
} // namespace

at::Tensor softmax(
    const at::Tensor &self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype)
{
    auto result = [&]() {
        at::NoNamesGuard guard;
        at::Tensor converted = dtype.has_value() ? at_npu::native::custom_ops::npu_dtype_cast(self, dtype.value()) : self;
        return at::_softmax(converted, dim, false);
    }();
    at::namedinference::propagate_names(result, self);

    return result;
}

at::Tensor softmax(
    const at::Tensor &self,
    at::Dimname dim,
    c10::optional<at::ScalarType> dtype)
{
    return acl_op::softmax(self, dimname_to_position(self, dim), dtype);
}

at::Tensor _softmax(const at::Tensor &self, int64_t dim, bool half_to_float)
{
    at::Tensor result;
    if (half_to_float) {
        result = npu_preparation::apply_tensor(self, self.options().dtype(at::ScalarType::Float));
    } else {
        result = npu_preparation::apply_tensor(self);
    }

    c10::optional<at::ScalarType> dtype = result.scalar_type();
    at::ScalarType dst_type;
    if (dtype.has_value()) {
        dst_type = dtype.value();
    } else if (result.defined()) {
        dst_type = result.scalar_type();
    } else {
        dst_type = self.scalar_type();
    }

    at::Tensor self_cast = dst_type == self.scalar_type() ?
        self : at_npu::native::custom_ops::npu_dtype_cast(self, dst_type);
    softmax_out_nocheck(result, self_cast, dim);
    return result;
}

at::Tensor& _softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    at::Tensor& out)
{
    auto dst_type = half_to_float ? at::kFloat : self.scalar_type();
    npu_preparation::CheckOut(
        {self},
        out,
        npu_preparation::get_tensor_npu_format(out),
        dst_type,
        self.sizes());

    auto self_dtype = self.scalar_type();
    if (half_to_float) {
        TORCH_CHECK(self_dtype == at::kHalf, "conversion is supported for Half type only" + OPS_ERROR(ErrCode::TYPE));
    } else {
        TORCH_CHECK(at::isFloatingType(self_dtype), "_softmax_npu not implemented for '", toString(self_dtype),
            "'" + OPS_ERROR(ErrCode::NOT_SUPPORT));
    }

    at::Tensor self_cast = dst_type == self.scalar_type() ?
        self : at_npu::native::custom_ops::npu_dtype_cast(self, dst_type);
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        softmax_out_nocheck(contiguous_result, self_cast, dim);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        softmax_out_nocheck(out, self_cast, dim);
    }
    return out;
}
} // namespace acl_op
