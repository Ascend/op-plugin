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
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {
at::Tensor& linspace_npu_out_nocheck(
    at::Tensor& result,
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps)
{
    if (steps != 0) {
        if (steps == 1) {
            acl_op::fill_(result, start);
        } else {
            c10::SmallVector<int64_t, N> size_vec = {steps};
            at_npu::native::OpCommand cmd;
            cmd.Name("LinSpace")
                .Input(start, at::ScalarType::Float)
                .Input(end, at::ScalarType::Float)
                .Input(size_vec, at::ScalarType::Int)
                .Output(result)
                .Run();
        }
    }
    return result;
}
}  // namespace

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& out)
{
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative"
        + OPS_ERROR(ErrCode::VALUE));

    if (out.numel() != steps) {
        out.resize_({steps});
    }

    bool out_is_not_float = (out.dtype() != at::kFloat) ? true : false;

    at::Tensor out_cast = out;
    if (out_is_not_float) {
        out_cast = at_npu::native::custom_ops::npu_dtype_cast(out, at::kFloat);
    }

    if (!npu_utils::check_match(&out_cast)) {
        at::Tensor contiguous_out = npu_utils::format_contiguous(out_cast);
        linspace_npu_out_nocheck(contiguous_out, start, end, steps);
        npu_utils::format_fresh_view(out_cast, contiguous_out);
    } else {
        linspace_npu_out_nocheck(out_cast, start, end, steps);
    }

    if (out_is_not_float) {
        out_cast = at_npu::native::custom_ops::npu_dtype_cast(out_cast, out.scalar_type());
        out.copy_(out_cast);
    } else {
        out = out_cast;
    }

    return out;
}

at::Tensor linspace(
    const at::Scalar& start,
    const at::Scalar& end,
    int64_t steps,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    TORCH_CHECK(steps >= 0, "number of steps must be non-negative"
        + OPS_ERROR(ErrCode::VALUE));
    auto device_value = c10::device_or_default(device);
    at::TensorOptions option = c10::TensorOptions()
        .dtype(dtype).layout(layout).device(device_value).pinned_memory(pin_memory);

    at::Tensor result = npu_preparation::apply_tensor_with_format({steps}, option, ACL_FORMAT_ND);
    at::Tensor result_cast = result;

    bool result_is_not_float = (result.dtype() != at::kFloat) ? true : false;
    if (result_is_not_float) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result, at::kFloat);
    }

    linspace_npu_out_nocheck(result_cast, start, end, steps);

    if (result_is_not_float) {
        result_cast = at_npu::native::custom_ops::npu_dtype_cast(result_cast, result.scalar_type());
    }
    result = result_cast;
    return result;
}
}  // namespace acl_op
