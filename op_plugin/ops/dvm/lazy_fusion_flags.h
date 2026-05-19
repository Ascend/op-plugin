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

#ifndef DVM_LAZY_FUSION_FLAGS_H
#define DVM_LAZY_FUSION_FLAGS_H

#include <cstdint>
#include <string>
#include <vector>

namespace lazy_fusion {

// Optimization level. Op-level promotion rule:
//   kO1 — conservative set (elementwise / activations / where /
//         batch-norm forward / foreach). Same set as the historical "level 1".
//   kO2 — kO1 plus the heavier ops (matmul / mm / bmm / addmm / sum /
//         native_batch_norm_backward / npu_swiglu) whose payoff depends on
//         workload. Same as historical "level 2".
// `TORCH_NPU_LAZY_FUSION=True` defaults to kO2 (enable everything). Setting
// `level=O1` / `level=O2` via the env flag list is internal-debug only.
enum class Level : int {
  kO1 = 1,
  kO2 = 2,
};

class LazyFusionFlags {
 public:
  LazyFusionFlags();
  ~LazyFusionFlags() = default;

  // Public knobs.
  bool enabled{false};
  Level level{Level::kO2};

  // Internal debug knobs (not advertised in user-facing docs).
  bool dump_as_text{false};
  std::string dump_dir{"./lazy_fusion_dump"};
  bool synchronize{false};
  bool online_tuning{false};
  std::vector<std::string> disable_ops;
  std::vector<std::string> enable_ops;
  std::vector<std::string> enable_ops_only;
};
}  // namespace lazy_fusion
#endif  // DVM_LAZY_FUSION_FLAGS_H
