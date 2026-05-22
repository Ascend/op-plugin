// Copyright (c) 2023 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
//
// aclnn Var/Std/VarMean/StdMean take int64 correction only. Use that path only when the scalar is
// exactly an integer in int64 range (PyTorch allows fractional ddof).

#ifndef OP_PLUGIN_UTILS_STD_VAR_CORRECTION_UTILS_H_
#define OP_PLUGIN_UTILS_STD_VAR_CORRECTION_UTILS_H_

#include <cmath>
#include <limits>

#include <c10/core/Scalar.h>
#include <c10/util/Optional.h>

inline bool correction_fits_aclnn_int64(const c10::optional<c10::Scalar> &correction)
{
    if (!correction.has_value()) {
        return true;
    }
    const c10::Scalar &s = correction.value();
    if (s.isIntegral(true)) {
        return true;
    }
    if (!s.isFloatingPoint()) {
        return false;
    }
    const double v = s.toDouble();
    if (!std::isfinite(v)) {
        return false;
    }
    const double t = std::trunc(v);
    if (v != t) {
        return false;
    }
    if (v > static_cast<double>(std::numeric_limits<int64_t>::max()) ||
        v < static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return false;
    }
    return true;
}

#endif // OP_PLUGIN_UTILS_STD_VAR_CORRECTION_UTILS_H_
