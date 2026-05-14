// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Mirrors aten/src/ATen/native/BucketizationUtils.h searchsorted_pre_check so NPU raises
// the same RuntimeError messages as CPU/CUDA (not CANN parameter errors).

#ifndef OP_PLUGIN_UTILS_SEARCHSORTED_VALIDATE_UTIL_H_
#define OP_PLUGIN_UTILS_SEARCHSORTED_VALIDATE_UTIL_H_

#include <climits>
#include <string>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

namespace op_plugin {

inline bool searchsorted_dims_matched_before_last_dim(const at::Tensor &boundaries, const at::Tensor &input) {
    if (boundaries.dim() != input.dim()) {
        return false;
    }
    const auto &dims_bd = boundaries.sizes();
    const auto &dims_in = input.sizes();
    for (int64_t dim = 0; dim + 1 < boundaries.dim(); ++dim) {
        if (dims_bd[dim] != dims_in[dim]) {
            return false;
        }
    }
    return true;
}

/// Full searchsorted pre_check for Tensor values; `output` may be null to skip out-dtype checks (scalar path).
inline void searchsorted_pre_check_npu(const at::Tensor &boundaries, const at::Tensor &input, const at::Tensor *output,
    bool out_int32, bool right, const c10::optional<c10::string_view> &side_opt,
    const c10::optional<at::Tensor> &sorter_opt) {
    if (side_opt.has_value()) {
        c10::string_view side = *side_opt;
        TORCH_CHECK(side == "left" || side == "right",
            "torch.searchsorted(): side can only be 'left' or 'right' but got ", std::string(side.data(), side.size()));

        TORCH_CHECK(!right || side == "right",
            "torch.searchsorted(): side and right can't be set to opposites, got side of ",
            std::string(side.data(), side.size()), " while right was True");
    }

    TORCH_CHECK(boundaries.device() == input.device(),
        "torch.searchsorted(): boundaries and input value tensors should have same device type, but got "
        "boundaries tensor device type ",
        boundaries.device(), " and input value tensor device type ", input.device());

    if (sorter_opt.has_value()) {
        const at::Tensor &sorter = *sorter_opt;
        TORCH_CHECK(
            sorter.defined(), "torch.searchsorted(): optional sorter was set but the sorter tensor is undefined");

        TORCH_CHECK(sorter.device() == boundaries.device(),
            "torch.searchsorted(): sorter and boundary tensors should have same device type, but got sorter tensor "
            "device type ",
            sorter.device(), " and input value tensor device type ", boundaries.device());

        TORCH_CHECK(sorter.sizes() == boundaries.sizes(),
            "torch.searchsorted(): boundary and sorter must have the same size, but got boundary tensor ",
            boundaries.sizes(), "and got sorter tensor ", sorter.sizes());

        TORCH_CHECK(sorter.scalar_type() == at::ScalarType::Long,
            "torch.searchsorted(): sorter must be a tensor of long dtype but got dtype ", sorter.scalar_type());

        if (sorter.numel() > 0) {
            auto minmax = sorter.aminmax();
            int64_t vmin = std::get<0>(minmax).item().toLong();
            int64_t vmax = std::get<1>(minmax).item().toLong();
            TORCH_CHECK(vmin >= 0 && vmax < sorter.sizes().back(), "torch.searchsorted(): sorter index out of range");
        }
    }

    TORCH_CHECK(input.dim() > 0 || (input.dim() == 0 && input.numel() == 1 && boundaries.dim() == 1),
        "torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we "
        "got boundaries tensor dim(",
        boundaries.dim(), ") and input value's dim(", input.dim(), ") numel(", input.numel(), ")");

    // Python `searchsorted(boundaries, 1)` may bind as a Tensor overload with shape [1] instead of a true
    // rank-0 scalar; CPU raises the scalar/boundaries-dim error, not "first N-1 dimensions...".
    if (boundaries.dim() != 1 && input.numel() == 1 && input.dim() != boundaries.dim()) {
        TORCH_CHECK(false,
            "torch.searchsorted(): input value can be a scalar only when boundaries tensor dimension is 1, but we "
            "got boundaries tensor dim(",
            boundaries.dim(), ") and input value's dim(", input.dim(), ") numel(", input.numel(), ")");
    }

    TORCH_CHECK(boundaries.dim() != 0,
        "torch.searchsorted(): boundaries tensor should have positive dimension, but got 0 dimension");

    TORCH_CHECK(boundaries.dim() == 1 || searchsorted_dims_matched_before_last_dim(boundaries, input),
        "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries "
        "tensor and input value tensor must match, but we got boundaries tensor ",
        boundaries.sizes(), " and input value tensor ", input.sizes());

    if (output != nullptr) {
        at::ScalarType output_dtype = output->scalar_type();
        TORCH_CHECK(
            (output_dtype == at::ScalarType::Long && !out_int32) || (output_dtype == at::ScalarType::Int && out_int32),
            "torch.searchsorted(): output tensor's dtype is wrong, it can only be Int(int32) or Long(int64) "
            "depending on whether out_int32 flag is True, but we got output tensor's dtype ",
            output_dtype, " and out_int32 flag is ", (out_int32 ? "True" : "False"));
    }

    if (out_int32) {
        TORCH_CHECK(boundaries.sizes().back() < INT_MAX,
            "torch.searchsorted(): the size of boundaries' last dimension should be less than ", INT_MAX,
            ", but we got ", boundaries.sizes().back());
    }
}

/// Functional Tensor overload: `out` is created after new_params in generated op_api, so validate everything
/// except output dtype here (matches CPU before aclnn). Tensor_out uses searchsorted_validate_tensor_out_op.
inline int searchsorted_validate_core_no_output(const at::Tensor &sorted_sequence, const at::Tensor &self,
    bool out_int32, bool right, const c10::optional<c10::string_view> &side_opt,
    const c10::optional<at::Tensor> &sorter_opt) {
    searchsorted_pre_check_npu(sorted_sequence, self, nullptr, out_int32, right, side_opt, sorter_opt);
    return 0;
}

/// Tensor_out / same shapes as aclnnSearchSorted. Returns 0 for use in yaml new_params.
inline int searchsorted_validate_tensor_out_op(const at::Tensor &sorted_sequence, const at::Tensor &self,
    const at::Tensor &out, bool out_int32, bool right, const c10::optional<c10::string_view> &side_opt,
    const c10::optional<at::Tensor> &sorter_opt) {
    searchsorted_pre_check_npu(sorted_sequence, self, &out, out_int32, right, side_opt, sorter_opt);
    return 0;
}

/// Scalar value: materialize wrapped-number tensor like aten (no output-dtype check; out is created by kernel).
inline int searchsorted_validate_scalar_op(const at::Tensor &sorted_sequence, const c10::Scalar &self, bool out_int32,
    bool right, const c10::optional<c10::string_view> &side_opt, const c10::optional<at::Tensor> &sorter_opt) {
    at::Tensor input_t = at::empty({}, sorted_sequence.options());
    input_t.fill_(self);
    input_t.unsafeGetTensorImpl()->set_wrapped_number(true);
    searchsorted_pre_check_npu(sorted_sequence, input_t, nullptr, out_int32, right, side_opt, sorter_opt);
    return 0;
}

} // namespace op_plugin

#endif // OP_PLUGIN_UTILS_SEARCHSORTED_VALIDATE_UTIL_H_
