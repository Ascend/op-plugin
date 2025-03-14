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
using npu_compile_type = at_npu::native::CompileType;
using npu_utils = at_npu::native::NpuUtils;

namespace {
inline bool allIntegral(std::initializer_list<std::reference_wrapper<at::Scalar>> values)
{
    for (at::Scalar &value : values) {
        if (!value.isIntegral(true)) {
            return false;
        }
    }
    return true;
}

at::Tensor &arange_out_npu_nocheck(at::Tensor &result, at::Scalar start, at::Scalar end, at::Scalar step)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Range")
        .Input(start, result.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(end, result.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Input(step, result.scalar_type(), npu_compile_type::MEMORY_HOST_COMPILE_DEPENDENT)
        .Output(result)
        .Run();
    return result;
}
} // namespace

at::Tensor arange(const at::Scalar &start, const at::Scalar &end, const at::Scalar &step,
                  c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype).device(device).layout(layout).pinned_memory(pin_memory);
    bool is_start_neq_end = true;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, c10::typeMetaToScalarType(option.dtype()), "arange_npu_ops", [&]() {
            using accscalar_type = at::acc_type<scalar_t, false>;
            auto start_value = start.to<accscalar_type>();
            auto end_value = end.to<accscalar_type>();
            auto step_value = step.to<accscalar_type>();

            TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero" + OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                            ((step_value < 0) && (end_value <= start_value)),
                        "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));
            at::Scalar start_opt = start;
            at::Scalar end_opt = end;
            at::Scalar step_opt = step;
            bool set_to_integral_dtype = !option.has_dtype() && allIntegral({start_opt, end_opt, step_opt});
            if (set_to_integral_dtype) {
                option = option.dtype(at::kLong);
            }
            is_start_neq_end = (start_value > end_value || start_value < end_value);
        });

    if (!is_start_neq_end) {
        at::Tensor result_empty = npu_preparation::apply_tensor_with_format({0}, option, ACL_FORMAT_ND);
        return result_empty;
    }

    auto output_size = op_infer::infersize_arange(start, end, step, c10::typeMetaToScalarType(option.dtype()));
    bool is_half = option.dtype() == at::kHalf || option.dtype() == at::kBFloat16;
    at::Tensor result =
        is_half ? npu_preparation::apply_tensor_with_format(output_size, option.dtype(at::kFloat), ACL_FORMAT_ND)
                : npu_preparation::apply_tensor_with_format(output_size, option, ACL_FORMAT_ND);
    arange_out_npu_nocheck(result, start, end, step);
    if (is_half) {
        result = result.to(option.dtype());
    }
    return result;
}

at::Tensor arange(const at::Scalar &start, const at::Scalar &end, c10::optional<at::ScalarType> dtype,
                  c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                  c10::optional<bool> pin_memory)
{
    const at::Scalar step = 1;
    return acl_op::arange(start, end, step, dtype, layout, device, pin_memory);
}


at::Tensor arange(const at::Scalar &end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    const at::Scalar start = 0;
    return acl_op::arange(start, end, dtype, layout, device, pin_memory);
}

at::Tensor &arange_out(const at::Scalar &start, const at::Scalar &end, const at::Scalar &step, at::Tensor &out)
{
    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "arange_out_npu_ops", [&]() {
        using accscalar_type = at::acc_type<scalar_t, false>;
        auto start_value = start.to<accscalar_type>();
        auto end_value = end.to<accscalar_type>();
        auto step_value = step.to<accscalar_type>();

        TORCH_CHECK(step_value != 0, "step must be nonzero" + OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                        ((step_value < 0) && (end_value <= start_value)),
                    "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));
    });

    auto output_size = op_infer::infersize_arange(start, end, step, out.scalar_type());
    if (out.numel() != output_size[0]) {
        TORCH_NPU_WARN("The out tensor size does not match the computed size, it will resize to computed size (",
                       output_size[0], ",).");
        out.resize_(output_size);
    }
    if (!npu_utils::check_match(&out)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(out);
        arange_out_npu_nocheck(contiguous_result, start, end, step);
        npu_utils::format_fresh_view(out, contiguous_result);
    } else {
        arange_out_npu_nocheck(out, start, end, step);
    }
    return out;
}

at::Tensor &arange_out(const at::Scalar &end, at::Tensor &out)
{
    const at::Scalar start = 0;
    const at::Scalar step = 1;
    return acl_op::arange_out(start, end, step, out);
}

at::Tensor _dim_arange(const at::Tensor &like, int64_t dim)
{
    c10::optional<at::ScalarType> dtype_opt(at::kInt);
    c10::optional<at::Layout> layout_opt(like.options().layout());
    c10::optional<at::Device> device_opt(like.options().device());
    c10::optional<bool> pin_memory_opt(like.options().pinned_memory());

    at::Tensor result = acl_op::arange(like.size(dim), dtype_opt, layout_opt, device_opt, pin_memory_opt);
    return result;
}
} // namespace acl_op
