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

#include <ATen/AccumulateType.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpAdapter.h"

namespace acl_op {
using npu_preparation = at_npu::native::OpPreparation;
using npu_utils = at_npu::native::NpuUtils;

namespace {

at::Tensor& range_out_nocheck(
    at::Tensor& result,
    at::Scalar start,
    at::Scalar end,
    at::Scalar step)
{
    // generate x assistant tensor
    int value = result.size(0);
    c10::SmallVector<int64_t, N> tmp_vector = {};
    for (int i = 0; i < value; i++) {
        tmp_vector.emplace_back(i);
    }

    at_npu::native::OpCommand cmd;
    cmd.Name("RangeD")
        .Input(tmp_vector, result.scalar_type())
        .Output(result)
        .Attr("start", start)
        .Attr("limit", end)
        .Attr("delta", step)
        .Run();

    return result;
}
} // namespace

at::Tensor range(
    const at::Scalar& start,
    const at::Scalar& end,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory)
{
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype).device(device).layout(layout).pinned_memory(pin_memory);
    return at::range(start, end, 1, option);
}

at::Tensor range(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                 c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                 c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype).device(device).layout(layout).pinned_memory(pin_memory);
    int64_t output_size = 0;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, c10::typeMetaToScalarType(option.dtype()), "range_npu_ops", [&]() {
            using accscalar_type = at::acc_type<scalar_t, false>;
            auto start_value = start.to<accscalar_type>();
            auto end_value = end.to<accscalar_type>();
            auto step_value = step.to<accscalar_type>();

            TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero", OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                            ((step_value < 0) && (end_value <= start_value)),
                        "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));
            output_size = static_cast<int64_t>((end_value - start_value) / step_value + 1);
        });

    at::Tensor result = npu_preparation::apply_tensor_with_format({output_size}, option, ACL_FORMAT_NCHW);
    return range_out_nocheck(result, start, end, step);
}

at::Tensor& range_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out)
{
    int64_t output_size = 0;
    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "range_out_npu_ops", [&]() {
        using accscalar_type = at::acc_type<scalar_t, false>;
        auto start_value = start.to<accscalar_type>();
        auto end_value = end.to<accscalar_type>();
        auto step_value = step.to<accscalar_type>();

        TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero" + OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                    ((step_value < 0) && (end_value <= start_value)),
                    "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));
        npu_preparation::CheckOut({}, out, ACL_FORMAT_NCHW, out.scalar_type(), out.sizes());

        output_size = static_cast<int64_t>((end_value - start_value) / step_value + 1);
    });

    if (out.numel() != output_size) {
        TORCH_NPU_WARN("The out tensor size does not match the computed size, it will resize to computed size (",
                       output_size, ",).");
        out.resize_({output_size});
    }

    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        range_out_nocheck(contiguous_result, start, end, step);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        range_out_nocheck(out, start, end, step);
    }
    return out;
}

} // namespace acl_op
