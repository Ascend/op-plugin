// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause

#ifndef OP_PLUGIN_UTILS_SEARCHSORTED_SIDE_UTIL_H_
#define OP_PLUGIN_UTILS_SEARCHSORTED_SIDE_UTIL_H_

#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

namespace op_plugin {

/// When `side` is set it overrides `right` (see torch.searchsorted(side=...)).
/// Invalid `side` and side/right conflicts are handled in searchsorted_pre_check_npu; call this after validate.
inline bool resolve_searchsorted_effective_right(bool right, const c10::optional<c10::string_view> &side_opt) {
    if (!side_opt.has_value()) {
        return right;
    }
    return *side_opt == "right";
}

} // namespace op_plugin

#endif // OP_PLUGIN_UTILS_SEARCHSORTED_SIDE_UTIL_H_
