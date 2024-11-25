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
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor range(const at::Scalar& start, const at::Scalar& end, c10::optional<at::ScalarType> dtype_opt,
                 c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                 c10::optional<bool> pin_memory_opt) {
  DO_COMPATIBILITY(aclnnRange, acl_op::range(start, end, dtype_opt, layout_opt, device_opt, pin_memory_opt));
  return op_api::range(start, end, 1, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor range(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
                 c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                 c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt)
{
    DO_COMPATIBILITY(aclnnRange, acl_op::range(start, end, step, dtype_opt, layout_opt, device_opt, pin_memory_opt));
    TORCH_CHECK(std::isfinite(start.toDouble()) && std::isfinite(end.toDouble()), "unsupported range: start -> end",
                OPS_ERROR(ErrCode::NOT_SUPPORT));
    c10::TensorOptions option =
        c10::TensorOptions().dtype(dtype_opt).device(device_opt).layout(layout_opt).pinned_memory(pin_memory_opt);
    int64_t output_size = 0;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::kHalf, at::kBFloat16, c10::typeMetaToScalarType(option.dtype()), "range_npu_nn", [&]() {
            using accscalar_type = at::acc_type<scalar_t, false>;
            auto start_value = start.to<accscalar_type>();
            auto end_value = end.to<accscalar_type>();
            auto step_value = step.to<accscalar_type>();

            TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero", OPS_ERROR(ErrCode::VALUE));
            TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                            ((step_value < 0) && (end_value <= start_value)),
                        "upper bound and larger bound inconsistent with step sign", OPS_ERROR(ErrCode::VALUE));

            output_size = static_cast<int64_t>((end_value - start_value) / step_value + 1);
        });

    at::Tensor result = npu_preparation::apply_tensor_without_format({output_size}, option);
    EXEC_NPU_CMD(aclnnRange, start, end, step, result);
    return result;
}

at::Tensor& range_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& result)
{
    DO_COMPATIBILITY(aclnnRange, acl_op::range_out(start, end, step, result));
    TORCH_CHECK(std::isfinite(start.toDouble()) && std::isfinite(end.toDouble()), "unsupported range: start -> end",
                OPS_ERROR(ErrCode::NOT_SUPPORT));

    int64_t output_size = 0;
    AT_DISPATCH_ALL_TYPES_AND2(at::kHalf, at::kBFloat16, result.scalar_type(), "range_out_npu_nn", [&]() {
        using accscalar_type = at::acc_type<scalar_t, false>;
        auto start_value = start.to<accscalar_type>();
        auto end_value = end.to<accscalar_type>();
        auto step_value = step.to<accscalar_type>();
        auto res_type = result.scalar_type();

        TORCH_CHECK(step_value > 0 || step_value < 0, "step must be nonzero", OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(((step_value > 0) && (end_value >= start_value)) ||
                        ((step_value < 0) && (end_value <= start_value)),
                    "upper bound and larger bound inconsistent with step sign", OPS_ERROR(ErrCode::VALUE));
        TORCH_CHECK(isFloatingType(res_type) || isIntegralType(res_type), "out datatype: ", res_type,
                    " unsupported datatype", OPS_ERROR(ErrCode::TYPE));
        npu_preparation::check_tensor({}, result, res_type, result.sizes());

        output_size = static_cast<int64_t>((end_value - start_value) / step_value + 1);
    });

    if (result.numel() != output_size) {
        TORCH_NPU_WARN("The out tensor size does not match the computed size, it will resize to computed size (",
                       output_size, ",).");
        result.resize_({output_size});
    }

    EXEC_NPU_CMD(aclnnRange, start, end, step, result);
    return result;
}
}
