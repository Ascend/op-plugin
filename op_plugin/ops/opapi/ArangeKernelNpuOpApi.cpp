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

#include <ATen/AccumulateType.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"
namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

// bool inputs are considered integral
static inline bool all_integral(std::initializer_list<std::reference_wrapper<at::Scalar>> l)
{
    for (at::Scalar& s : l) {
        if (!s.isIntegral(true)) {
        return false;
        }
    }
    return true;
}

static at::Tensor& arange_out_op_api(at::Scalar start, at::Scalar end, at::Scalar step, at::Tensor& result)
{
    EXEC_NPU_CMD(aclnnArange, start, end, step, result);
    return result;
}

at::Tensor arange(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                  c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnArange, acl_op::arange(start, end, step, dtype, layout, device, pin_memory));
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype).device(device).layout(layout).pinned_memory(pin_memory);

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, c10::typeMetaToScalarType(option.dtype()), "arange_npu_nn", [&]() {
            using accscalar_type = at::acc_type<scalar_t, false>;
            auto start_value = start.to<accscalar_type>();
            auto end_value = end.to<accscalar_type>();
            auto step_value = step.to<accscalar_type>();

            TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero", OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                            ((step_value < 0) && (end_value <= start_value)),
                        "upper bound and larger bound inconsistent with step sign", OPS_ERROR(ErrCode::VALUE));
        });

    at::Scalar start_opt = start;
    at::Scalar end_opt = end;
    at::Scalar step_opt = step;
    bool set_to_integral_dtype = !option.has_dtype() && all_integral({start_opt, end_opt, step_opt});
    if (set_to_integral_dtype) {
        option = option.dtype(at::ScalarType::Long);
    }

    auto output_size = op_infer::infersize_arange(start, end, step, c10::typeMetaToScalarType(option.dtype()));
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size, option);
    arange_out_op_api(start, end, step, result);

    return result;
}

at::Tensor arange(const at::Scalar& start, const at::Scalar& end, c10::optional<at::ScalarType> dtype,
                  c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                  c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnArange, acl_op::arange(start, end, dtype, layout, device, pin_memory));
    return op_api::arange(start, end, 1, dtype, layout, device, pin_memory);
}

at::Tensor arange(const at::Scalar& end, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                  c10::optional<at::Device> device, c10::optional<bool> pin_memory)
{
    DO_COMPATIBILITY(aclnnArange, acl_op::arange(end, dtype, layout, device, pin_memory));
    return op_api::arange(0, end, dtype, layout, device, pin_memory);  // start = 0
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnArange, acl_op::arange_out(start, end, step, out));

    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, out.scalar_type(), "arange_out_npu_nn", [&]() {
        using accscalar_type = at::acc_type<scalar_t, false>;
        auto start_value = start.to<accscalar_type>();
        auto end_value = end.to<accscalar_type>();
        auto step_value = step.to<accscalar_type>();

        TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero", OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                        ((step_value < 0) && (end_value <= start_value)),
                    "upper bound and larger bound inconsistent with step sign", OPS_ERROR(ErrCode::VALUE));
    });

    auto output_size = op_infer::infersize_arange(start, end, step, out.scalar_type());
    if (out.numel() != output_size[0]) {
        TORCH_NPU_WARN("The out tensor size does not match the computed size, it will resize to computed size (",
                       output_size[0], ",).");
        out.resize_(output_size);
    }
    arange_out_op_api(start, end, step, out);
    return out;
}

static at::Tensor& arange_start_end_out(at::Scalar start, at::Scalar end, at::Tensor& result)
{
    at::Scalar step = 1;
    return op_api::arange_out(start, end, step, result);
}

at::Tensor& arange_out(const at::Scalar& end, at::Tensor& out)
{
    DO_COMPATIBILITY(aclnnArange, acl_op::arange_out(end, out));
    return arange_start_end_out(0, end, out);
}
}
