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
                  c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                  c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt)
{
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);

    float start_value = op_plugin::utils::get_scalar_float_value(start);
    float end_value = op_plugin::utils::get_scalar_float_value(end);
    float step_value = op_plugin::utils::get_scalar_float_value(step);

    TORCH_CHECK(step_value != 0, "step must be nonzero" + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
        "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));
    at::Scalar start_opt = start;
    at::Scalar end_opt = end;
    at::Scalar step_opt = step;
    bool set_to_integral_dtype = !option.has_dtype() && allIntegral({start_opt, end_opt, step_opt});
    // check start == end
    if (set_to_integral_dtype) {
        option = option.dtype(at::kLong);
    }
    at::Tensor result_check = npu_preparation::apply_tensor_with_format({0}, option, ACL_FORMAT_ND);
    if (start_value == end_value) {
        return result_check;
    }

    auto output_size = op_infer::infersize_arange(start, end, step);
    bool is_half = option.dtype() == at::kHalf || option.dtype() == at::kBFloat16;
    at::Tensor result =
        is_half ? npu_preparation::apply_tensor_with_format(output_size, option.dtype(at::kFloat), ACL_FORMAT_ND) :
                  npu_preparation::apply_tensor_with_format(output_size, option, ACL_FORMAT_ND);
    arange_out_npu_nocheck(result, start, end, step);
    if (is_half) {
        result = result.to(option.dtype());
    }
    return result;
}

at::Tensor arange(const at::Scalar &start, const at::Scalar &end, c10::optional<at::ScalarType> dtype_opt,
                  c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                  c10::optional<bool> pin_memory_opt)
{
    const at::Scalar step = 1;
    return acl_op::arange(start, end, step, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}


at::Tensor arange(const at::Scalar &end, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                  c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt)
{
    const at::Scalar start = 0;
    return acl_op::arange(start, end, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor &arange_out(const at::Scalar &start, const at::Scalar &end, const at::Scalar &step, at::Tensor &result)
{
    float start_value = op_plugin::utils::get_scalar_float_value(start);
    float end_value = op_plugin::utils::get_scalar_float_value(end);
    float step_value = op_plugin::utils::get_scalar_float_value(step);
    TORCH_CHECK(step_value != 0, "step must be nonzero" + OPS_ERROR(ErrCode::VALUE));
    TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) || ((step_value < 0) && (end_value <= start_value)),
        "upper bound and larger bound inconsistent with step sign" + OPS_ERROR(ErrCode::VALUE));

    auto output_size = op_infer::infersize_arange(start, end, step);
    result.resize_(output_size);
    if (!npu_utils::check_match(&result)) {
        at::Tensor contiguous_result = npu_utils::format_contiguous(result);
        arange_out_npu_nocheck(contiguous_result, start, end, step);
        npu_utils::format_fresh_view(result, contiguous_result);
    } else {
        arange_out_npu_nocheck(result, start, end, step);
    }
    return result;
}

at::Tensor &arange_out(const at::Scalar &end, at::Tensor &result)
{
    const at::Scalar start = 0;
    const at::Scalar step = 1;
    return acl_op::arange_out(start, end, step, result);
}

at::Tensor _dim_arange(const at::Tensor &self, int64_t dim)
{
    c10::optional<at::ScalarType> dtype_opt(at::kInt);
    c10::optional<at::Layout> layout_opt(self.options().layout());
    c10::optional<at::Device> device_opt(self.options().device());
    c10::optional<bool> pin_memory_opt(self.options().pinned_memory());

    at::Tensor result = acl_op::arange(self.size(dim), dtype_opt, layout_opt, device_opt, pin_memory_opt);
    return result;
}
} // namespace acl_op
