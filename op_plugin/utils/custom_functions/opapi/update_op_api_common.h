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

#ifndef TORCHNPU_TORCH_NPU_CUSTOM_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CUSTOM_OPS_OP_API_PTA_COMMON_H_

#include "op_plugin/utils/op_api_common.h"

#define EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD_V1(aclnn_api, workspace_addr, workspace_size, ...)                         \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD", "1", __VA_ARGS__);              \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        at_npu::native::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        uint64_t fake_workspace_size = 0;                                                                              \
        uint64_t *workspace_size_addr = &fake_workspace_size;                                                          \
        auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                     \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                               \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {              \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");                                                    \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
    } while (false)

#define EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD_V2(aclnn_api, workspace_addr, workspace_size, ...)                         \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");                             \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    "not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD", "2", __VA_ARGS__);              \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        auto acl_call = [workspace_addr, workspace_size, copied_params, acl_stream]()->int {                           \
            uint64_t fake_workspace_size = 0;                                                                          \
            uint64_t *workspace_size_addr = &fake_workspace_size;                                                      \
            aclOpExecutor *executor = nullptr;                                                                         \
            aclOpExecutor **executor_addr = &executor;                                                                 \
            InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
            UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);        \
            InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                      \
                reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                \
            SetPTACacheHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);       \
            if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                    \
                initPTACacheThreadLocalFunc();                                                                         \
                setPTAHashKeyFunc(nullptr, 0);                                                                         \
            }                                                                                                          \
            at_npu::native::SetDeterministic();                                                                        \
            if (initMemFunc) {                                                                                         \
                initMemFunc(nullptr, false);                                                                           \
            }                                                                                                          \
            auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                 \
            auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);                \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                      \
            NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                           \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");                                                    \
            ReleaseConvertTypes(converted_params);                                                                     \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                          \
            if (releaseMemFunc) {                                                                                      \
                releaseMemFunc(nullptr, false);                                                                        \
            }                                                                                                          \
            if (unInitMemFunc) {                                                                                       \
                unInitMemFunc(nullptr, false);                                                                         \
            }                                                                                                          \
            UnInitCacheThreadLocal();                                                                                  \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                                   \
    } while (false)

#define EXEC_GET_MAX_WORKSPACE_CMD(aclnn_api, ...)                                                                     \
    [](const char *apiName, auto &...args)->auto {                                                                     \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetMaxWorkspaceSize");               \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr, #aclnn_api "GetMaxWorkspaceSize", " not in ",                 \
                    GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.", OPS_ERROR(ErrCode::PTR));             \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
        }                                                                                                              \
        at_npu::native::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);                             \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                               \
        ReleaseConvertTypes(converted_params);                                                                         \
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                              \
        if (releaseMemFunc) {                                                                                          \
            releaseMemFunc(nullptr, false);                                                                            \
        }                                                                                                              \
        if (unInitMemFunc) {                                                                                           \
            unInitMemFunc(nullptr, false);                                                                             \
        }                                                                                                              \
        UnInitCacheThreadLocal();                                                                                      \
        return workspace_size;                                                                                         \
    }(#aclnn_api, __VA_ARGS__)


#define EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD(aclnn_api, workspace_addr, workspace_size, ...)                            \
    do {                                                                                                               \
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();                   \
        if (task_queue_enable == 2) {                                                                                  \
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD_V2(aclnn_api, workspace_addr, workspace_size, __VA_ARGS__);            \
        } else {                                                                                                       \
            EXEC_UPDATE_NPU_NO_FORMAT_CHECK_CMD_V1(aclnn_api, workspace_addr, workspace_size, __VA_ARGS__);            \
        }                                                                                                              \
    } while (false)


#endif //  TORCHNPU_TORCH_NPU_CUSTOM_OPS_OP_API_PTA_COMMON_H_
