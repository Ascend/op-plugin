// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Aligns with aten/src/ATen/native/BucketizationUtils.h (searchsorted_maybe_trim_input_tensors).

#ifndef OP_PLUGIN_UTILS_SEARCHSORTED_WARN_UTIL_H_
#define OP_PLUGIN_UTILS_SEARCHSORTED_WARN_UTIL_H_

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

namespace op_plugin {

/// Row-major contiguous check aligned with dense strided layout (matches TensorImpl contiguous semantics).
/// Some backends may report is_contiguous() true while strides still require an explicit copy for aclnn; this
/// catches permute().to(...) cases that CPU flags in searchsorted_maybe_trim_input_tensors.
inline bool searchsorted_tensor_is_row_major_contiguous(const at::Tensor &t) {
    if (!t.defined() || t.numel() == 0) {
        return true;
    }
    if (t.layout() != c10::Layout::Strided) {
        return t.is_contiguous();
    }
    const int64_t dim = t.dim();
    if (dim == 0) {
        return true;
    }
    int64_t z = 1;
    for (int64_t d = dim - 1; d >= 0; --d) {
        const int64_t size_d = t.size(d);
        if (size_d != 1) {
            if (t.stride(d) != z) {
                return false;
            }
            if (size_d == 0) {
                return true;
            }
        }
        z *= size_d;
    }
    return true;
}

/// Tensor, Tensor overload: warn once per process when inputs may need a contiguous copy (matches CPU/CUDA).
inline int warn_if_searchsorted_inputs_noncontiguous(
    const at::Tensor &sorted_sequence, const at::Tensor &self, const c10::optional<at::Tensor> &sorter_opt) {
    if (!searchsorted_tensor_is_row_major_contiguous(self)) {
        TORCH_WARN_ONCE(
            "torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due "
            "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous input value "
            "tensor if possible. This message will only appear once per program.");
    }
    if (!searchsorted_tensor_is_row_major_contiguous(sorted_sequence)) {
        TORCH_WARN_ONCE(
            "torch.searchsorted(): boundary tensor is non-contiguous, this will lower the performance due "
            "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous boundary "
            "tensor if possible. This message will only appear once per program.");
    }
    if (sorter_opt.has_value()) {
        const at::Tensor &st = *sorter_opt;
        if (st.defined() && !searchsorted_tensor_is_row_major_contiguous(st)) {
            TORCH_WARN_ONCE(
                "torch.searchsorted(): sorter tensor is non-contiguous, this will lower the performance due "
                "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous sorter "
                "tensor if possible. This message will only appear once per program.");
        }
    }
    return 0;
}

/// Tensor, Scalar overload: only boundaries / sorter apply (scalar value is materialized separately).
inline int warn_if_searchsorted_scalar_inputs_noncontiguous(
    const at::Tensor &sorted_sequence, const c10::optional<at::Tensor> &sorter_opt) {
    if (!searchsorted_tensor_is_row_major_contiguous(sorted_sequence)) {
        TORCH_WARN_ONCE(
            "torch.searchsorted(): boundary tensor is non-contiguous, this will lower the performance due "
            "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous boundary "
            "tensor if possible. This message will only appear once per program.");
    }
    if (sorter_opt.has_value()) {
        const at::Tensor &st = *sorter_opt;
        if (st.defined() && !searchsorted_tensor_is_row_major_contiguous(st)) {
            TORCH_WARN_ONCE(
                "torch.searchsorted(): sorter tensor is non-contiguous, this will lower the performance due "
                "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous sorter "
                "tensor if possible. This message will only appear once per program.");
        }
    }
    return 0;
}

} // namespace op_plugin

#endif // OP_PLUGIN_UTILS_SEARCHSORTED_WARN_UTIL_H_
