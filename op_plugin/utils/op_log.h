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

#ifndef TORCHNPU_TORCH_NPU_UTILS_OP_LOG_H_
#define TORCHNPU_TORCH_NPU_UTILS_OP_LOG_H_

#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include <thread>

#include "torch_npu/csrc/logging/LogContext.h"

#include "op_plugin/utils/OpAdapter.h"
#include "op_plugin/utils/op_log_utils.h"


namespace op_plugin {
namespace logging {

extern std::shared_ptr<npu_logging::Logger> LOGGER;
extern thread_local int log_depth;

// Logging function for op execute with task queue.
#define OP_EXEC_LOG_WITH_TASK_QUEUE(op_name, exec_cmd, task_queue, acl_stream, ...)                                 \
    do {                                                                                                               \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::INFO &&                          \
            op_plugin::logging::log_depth == 0) {                                                                      \
            op_plugin::logging::log_depth += 1;                                                                        \
            int64_t stream_id = 0;                                                                                     \
            if (c10_npu::acl::IsExistRtGetStreamId()) {                                                                \
                int32_t stream_ptr = 0;                                                                                \
                if (c10_npu::acl::AclrtStreamGetId(acl_stream, &stream_ptr) != ACL_ERROR_NONE) {                       \
                    stream_id = -1;                                                                                    \
                } else {                                                                                               \
                    stream_id = stream_ptr;                                                                            \
                }                                                                                                      \
            } else {                                                                                                   \
                stream_id = reinterpret_cast<int64_t>(acl_stream);                                                     \
            }                                                                                                          \
            const uint32_t aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);           \
            const uint32_t aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);         \
            op_plugin::logging::LOGGER->long_info(                                                                     \
                "%s %s stream_id=%ld, aic_num=%u aiv_num=%u with task_queue = %s%s",                                  \
                op_name, exec_cmd, stream_id, aic_num, aiv_num, task_queue,                                           \
                op_plugin::logging::generate_log_infos(#__VA_ARGS__, ##__VA_ARGS__).c_str());                            \
            op_plugin::logging::log_depth -= 1;                                                                        \
        }                                                                                                              \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::DEBUG &&                         \
            op_plugin::logging::log_depth == 0) {                                                                      \
            op_plugin::logging::log_depth += 1;                                                                        \
            int64_t stream_id = 0;                                                                                     \
            if (c10_npu::acl::IsExistRtGetStreamId()) {                                                                \
                int32_t stream_ptr = 0;                                                                                \
                if (c10_npu::acl::AclrtStreamGetId(acl_stream, &stream_ptr) != ACL_ERROR_NONE) {                       \
                    stream_id = -1;                                                                                    \
                } else {                                                                                               \
                    stream_id = stream_ptr;                                                                            \
                }                                                                                                      \
            } else {                                                                                                   \
                stream_id = reinterpret_cast<int64_t>(acl_stream);                                                     \
            }                                                                                                          \
            const uint32_t aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);           \
            const uint32_t aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);         \
            at_npu::native::OpCommand cmd;                                                                             \
            op_plugin::logging::LOGGER->long_info(                                                                     \
                "%s %s stream_id=%ld, aic_num=%u aiv_num=%u with task_queue = %s%s",                                  \
                op_name, exec_cmd, stream_id, aic_num, aiv_num, task_queue,                                           \
                op_plugin::logging::generate_log_infos(#__VA_ARGS__, ##__VA_ARGS__).c_str());                          \
            op_plugin::logging::LOGGER->long_debug("%s %s",                                                            \
                op_name, op_plugin::logging::generate_debug_log_infos(#__VA_ARGS__, ##__VA_ARGS__).c_str());             \
            op_plugin::logging::log_depth -= 1;                                                                        \
        }                                                                                                              \
    } while (0);

// Logging function for op execute.
#define OP_EXEC_LOG(op_name, exec_cmd, ...)                                                                                                 \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::INFO && op_plugin::logging::log_depth == 0) {         \
            op_plugin::logging::log_depth += 1;                                                                                             \
            op_plugin::logging::LOGGER->long_info(__FILE__, __LINE__, "%s %s with%s",                                                       \
                #op_name, exec_cmd, op_plugin::logging::generate_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());                             \
            op_plugin::logging::log_depth -= 1;                                                                                             \
        }                                                                                                                                   \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::DEBUG && op_plugin::logging::log_depth == 0) {        \
            op_plugin::logging::log_depth += 1;                                                                                             \
            op_plugin::logging::LOGGER->long_info(__FILE__, __LINE__, "%s %s with%s",                                                       \
                #op_name, exec_cmd, op_plugin::logging::generate_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());                             \
            op_plugin::logging::LOGGER->long_debug(__FILE__, __LINE__, "%s %s",                                                             \
                #op_name, op_plugin::logging::generate_debug_log_infos(#__VA_ARGS__, __VA_ARGS__).c_str());                                 \
            op_plugin::logging::log_depth -= 1;                                                                                             \
        }                                                                                                                                   \
    } while (0);

// Common op_plugin logging function.
#define OP_LOG_DEBUG(fmt, ...)                                                                                                              \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::DEBUG) {                                              \
            op_plugin::logging::LOGGER->debug(__FILE__, __LINE__, "%s", #fmt);                                                              \
        }                                                                                                                                   \
    } while (0);

#define OP_LOG_INFO(fmt, ...)                                                                                                               \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::INFO) {                                               \
            op_plugin::logging::LOGGER->info(__FILE__, __LINE__, "%s", #fmt);                                                               \
        }                                                                                                                                   \
    } while (0);

#define OP_LOG_WARNING(fmt, ...)                                                                                                            \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::WARNING) {                                            \
            op_plugin::logging::LOGGER->warn(__FILE__, __LINE__, "%s", #fmt);                                                               \
        }                                                                                                                                   \
    } while (0);

#define OP_LOG_ERROR(fmt, ...)                                                                                                              \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::ERROR) {                                              \
            op_plugin::logging::LOGGER->error(__FILE__, __LINE__, "%s", #fmt);                                                              \
        }                                                                                                                                   \
    } while (0);

#define OP_LOG_CRITICAL(fmt, ...)                                                                                                           \
    do {                                                                                                                                    \
        if (op_plugin::logging::LOGGER->getAllowLevel() == npu_logging::LoggingLevel::CRITICAL) {                                           \
            op_plugin::logging::LOGGER->critical(__FILE__, __LINE__, "%s", #fmt);                                                           \
        }                                                                                                                                   \
    } while (0);

}  // namespace logging
}  // namespace op_plugin

#endif //  TORCHNPU_TORCH_NPU_UTILS_OP_LOG_H_
