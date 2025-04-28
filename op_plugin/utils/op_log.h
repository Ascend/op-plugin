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

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>

#include "torch_npu/csrc/logging/LogContext.h"

#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_log_utils.h"


namespace op_plugin {
namespace logging {

static std::shared_ptr<npu_logging::Logger> LOGGER = npu_logging::logging().getLogger("torch_npu.op_plugin");

#define OP_EXEC_LOG(aclnn_api, exec_cmd, task_queue, ...)                                                                                   \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::INFO) {                                               \
            op_plugin::logging::LOGGER->info("%s %s with task_queue = %s%s",                                                                \
                         aclnn_api, exec_cmd, task_queue, op_plugin::logging::generate_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());       \
        }                                                                                                                                   \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::DEBUG) {                                              \
            op_plugin::logging::LOGGER->info("%s %s with task_queue = %s%s",                                                                \
                aclnn_api, exec_cmd, task_queue, op_plugin::logging::generate_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());                \
            op_plugin::logging::LOGGER->debug("%s %s",                                                                                      \
                aclnn_api, op_plugin::logging::generate_debug_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());                                \
        }                                                                                                                                   \
    } while (0);

}  // namespace utils
}  // namespace op_plugin
