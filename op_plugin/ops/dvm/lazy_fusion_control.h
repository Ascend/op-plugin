// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DVM_LAZY_FUSION_CONTROL_H
#define DVM_LAZY_FUSION_CONTROL_H

namespace lazy_fusion {
// Update a process-wide script-side gate, flushing at state transitions.
// Return the previous state so callers can restore it on context exit.
bool SetLazyFusionDisabled(bool disabled);
bool SetLazyFusionDumpEnabled(bool enabled);
}  // namespace lazy_fusion

#endif  // DVM_LAZY_FUSION_CONTROL_H
