// Copyright (c) 2026 Huawei Technologies Co., Ltd
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
#pragma once

// PyTorch 2.13 removed named-tensor support entirely (at::Dimname /
// at::DimnameList / the at::namedinference namespace / ATen/NamedTensorUtils.h).
// op-plugin still supports older PyTorch (v2.1..v2.12) where named tensors
// exist, so its named-tensor operator overloads are gated out of registration
// for v2.13+, but the now-dead code still references these symbols and must keep
// compiling. Forward to the real ATen header on < v2.13 and provide minimal
// stand-ins on >= v2.13.
//
// Include this instead of <ATen/NamedTensorUtils.h>.
// CAN REMOVE the >= v2.13 branch (and this header) when the minimum supported
// version is v2.13.
#include "op_plugin/utils/Version.h"

// The removed ATen/NamedTensorUtils.h used to pull in ATen/WrapDimUtils.h
// transitively; some kernels rely on that for at::maybe_wrap_dims(_n).
// WrapDimUtils.h is unrelated to named tensors and still exists, so include it
// unconditionally to keep this header a faithful drop-in replacement.
#include <ATen/WrapDimUtils.h>

#if !VERSION_BETWEEN(V2R13, VERSION_NEWEST)

#include <ATen/NamedTensorUtils.h>

#else

#include <vector>
#include <c10/util/ArrayRef.h>

namespace at {
struct Dimname {};
using DimnameList = c10::ArrayRef<at::Dimname>;

// RAII guard that disabled name propagation; a no-op once names are removed.
struct NoNamesGuard {};

// All Dimname / DimnameList operator overloads that need
// dimname_to_position / dimnames_to_positions are gated out on v2.13+ via
// #if !VERSION_BETWEEN(V2R13, VERSION_NEWEST), so those two helpers are not
// stubbed here. The remaining namedinference::* stubs are called from function
// bodies of non-named overloads (name-propagation calls sprinkled through
// matmul / addmm / bmm / cat / reduction / broadcast paths) and must exist as
// no-ops on v2.13+ to let those files compile.
namespace namedinference {
template <class... Args>
inline void propagate_names(Args &&...) {}
template <class... Args>
inline void propagate_names_if_nonempty(Args &&...) {}
template <class... Args>
inline void propagate_names_for_reduction(Args &&...) {}
template <class... Args>
inline ::std::vector<at::Dimname> propagate_names_for_addmv(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> propagate_names_for_addmm(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> compute_matmul_outnames(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> compute_bmm_outnames(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> compute_broadcast_outnames(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> compute_cdist_outnames(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> compute_cat_outnames(Args &&...) { return {}; }
template <class... Args>
inline ::std::vector<at::Dimname> broadcast_to_outnames(Args &&...) { return {}; }
template <class... Args>
inline bool are_names_equal(Args &&...) { return true; }
}  // namespace namedinference
}  // namespace at

#endif  // !VERSION_BETWEEN(V2R13, VERSION_NEWEST)
