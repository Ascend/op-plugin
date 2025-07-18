// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "op_log.h"

namespace op_plugin {
namespace logging {

std::shared_ptr<npu_logging::Logger> LOGGER = npu_logging::logging().getLogger("torch_npu.op_plugin");
thread_local int log_depth = 0;

}  // namespace utils
}  // namespace op_plugin
