// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_

#include "op_api_common_base.h"

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;
extern const std::vector<std::string> g_custom_lib_path;
extern const std::vector<std::string> g_default_custom_lib_path;
extern const std::vector<std::string> g_opApiSoFiles;
extern const std::vector<void *> g_opApiHandlers;

namespace {
constexpr int64_t MAX_DIM_NUM = 5;
constexpr int64_t NCL_DIM_NUM = 3;
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;
constexpr int64_t FP4_IN_INT8 = 2;
constexpr int64_t PENULTIMATE_DIM = 2;
}

std::string real_path(const std::string &path);
bool checkOwner(string cusLibPath);

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline const char *GetCustOpApiLibName(void)
{
    return "libcust_opapi.so";
}

inline void *GetOpApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetOpApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

#define GET_OP_API_FUNC_FROM_FEATURE_LIB(lib_handler, lib_name, api_name)                                              \
    do {                                                                                                               \
        static auto lib_handler = GetOpApiLibHandler((lib_name));                                                      \
        if ((lib_handler) != nullptr) {                                                                                \
            auto funcAddr = GetOpApiFuncAddrInLib((lib_handler), (lib_name), (api_name));                              \
            if (funcAddr != nullptr) {                                                                                 \
                return funcAddr;                                                                                       \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name);

bool check_aclnn_kernel_available(std::string aclnn_name);

uint64_t calc_hash_id();

#define DO_COMPATIBILITY(aclnn_api, originCallExpression)                                                              \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto isAclnnOnly = c10_npu::IsAclnnOnly();                                                        \
        if (getWorkspaceSizeFuncAddr == nullptr || opApiFuncAddr == nullptr) {                                         \
            if (isAclnnOnly) {                                                                                         \
                TORCH_CHECK(false, "Current device only support aclnn operators, but ",                                \
                    #aclnn_api, " or ", #aclnn_api, "GetWorkspaceSize not found", OPS_ERROR(ErrCode::NOT_SUPPORT));    \
            }                                                                                                          \
            ASCEND_LOGW("%s or %sGetWorkspaceSize not in %s, or %s not found. Will call %s", #aclnn_api, #aclnn_api,   \
                        GetOpApiLibName(), GetOpApiLibName(), #originCallExpression);                                  \
            return originCallExpression;                                                                               \
        }                                                                                                              \
    } while (0)

typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);
typedef aclOpExecutor *(*PTAGetExecCache)(uint64_t, uint64_t *);
typedef aclOpExecutor *(*PTAFindExecCache)(uint8_t *, size_t, uint64_t *);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef void (*SetPTACacheHashKey)(uint8_t *, size_t);
typedef bool (*CanUsePTACache)(const char *);
typedef void (*UnInitPTACacheThreadLocal)();

inline void UnInitCacheThreadLocal()
{
    static const auto unInitPTACacheThreadLocalAddr = GetOpApiFuncAddr("UnInitPTACacheThreadLocal");
    UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
        reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
    if (unInitPTACacheThreadLocalFunc) {
        unInitPTACacheThreadLocalFunc();
    }
}

// Check a tensor is on NPU and that a tensor with non-zero elements has allocated storage.
inline void CheckNpuTensorValid(const at::Tensor& at_tensor)
{
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
    TORCH_CHECK(at_tensor.numel() <= 0 || at_tensor.unsafeGetTensorImpl()->storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet.",
        OPS_ERROR(ErrCode::VALUE));
}

template <typename... Args> bool hit_cache(aclrtStream acl_stream, const char *aclnn_api, void *phrase2, Args &&...args)
{
    static const auto ptaGetExecCacheAddr = GetOpApiFuncAddr("PTAGetExecCache");
    static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
    static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");
    static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");
    PTAGetExecCache ptaGetExecCacheFunc = reinterpret_cast<PTAGetExecCache>(ptaGetExecCacheAddr);
    InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
        reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
    SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);
    CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);
    bool has_func = ptaGetExecCacheFunc && initPTACacheThreadLocalFunc && setPTAHashKeyFunc;
    bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(aclnn_api);
    if (!has_func || !can_use) {
        return false;
    }
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    initPTACacheThreadLocalFunc();
    g_hash_offset = 0;
    auto deterministic = at::globalContext().deterministicAlgorithms();
    if (c10_npu::is_core_control_enabled()) {
        auto aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
        auto aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
        add_param_to_buf(aic_num);
        add_param_to_buf(aiv_num);
    }
    auto device = c10_npu::current_device();
    add_param_to_buf(deterministic);
    add_param_to_buf(std::string(aclnn_api), args...);
    add_param_to_buf(device);
    add_param_to_buf((uintptr_t)acl_stream);
    uint64_t hashId = calc_hash_id();
    setPTAHashKeyFunc(hashId);
    aclOpExecutor *executor = ptaGetExecCacheFunc(hashId, workspace_size_addr);
    if (executor == nullptr) {
        return false;
    }
    void *workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }
    auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, phrase2]()->int {
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        NPU_CHECK_ERROR(api_ret, "call failed");
        return api_ret;
    };
    at_npu::native::OpCommand::RunOpApiV2(aclnn_api, acl_call);
    UnInitCacheThreadLocal();
    return true;
}

template <typename ...Ts>
bool hit_cache_v2(
    aclrtStream acl_stream, const char *aclnn_api, void *phrase2, const std::tuple<Ts...> &args, int* api_ret,
    bool deterministic_status, uint32_t aic_num, uint32_t aiv_num)
{
    static const auto ptaFindExecCacheAddr = GetOpApiFuncAddr("PTAFindExecCache");
    static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");
    static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");
    static const auto canUsePTACacheAddr = GetOpApiFuncAddr("CanUsePTACache");
    PTAFindExecCache ptaFindExecCacheFunc = reinterpret_cast<PTAFindExecCache>(ptaFindExecCacheAddr);
    InitPTACacheThreadLocal initPTACacheThreadLocalFunc =
        reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);
    SetPTACacheHashKey setPTACacheHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);
    CanUsePTACache canUsePTACacheFunc = reinterpret_cast<CanUsePTACache>(canUsePTACacheAddr);
    bool has_func = ptaFindExecCacheFunc && initPTACacheThreadLocalFunc && setPTACacheHashKeyFunc;
    bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(aclnn_api);
    if (!has_func || !can_use) {
        return false;
    }
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    initPTACacheThreadLocalFunc();
    g_hash_offset = 0;
    add_param_to_buf_v2(deterministic_status);
    if (aic_num != UINT32_MAX && aiv_num != UINT32_MAX) {
        add_param_to_buf_v2(aic_num);
        add_param_to_buf_v2(aiv_num);
    }
    add_param_to_buf_v2(std::string(aclnn_api));
    add_params_to_buf_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
    add_param_to_buf_v2((uintptr_t)acl_stream);
    if (g_hash_offset == g_hash_buf_max_size) {
        setPTACacheHashKeyFunc(nullptr, 0);
    } else {
        setPTACacheHashKeyFunc(reinterpret_cast<uint8_t *>(g_hash_buf), g_hash_offset);
    }
    aclOpExecutor *executor = ptaFindExecCacheFunc(reinterpret_cast<uint8_t *>(g_hash_buf),
        g_hash_offset, workspace_size_addr);
    if (executor == nullptr) {
        return false;
    }
    void *workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(phrase2);
    *api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    NPU_CHECK_ERROR(*api_ret, "call failed");
    UnInitCacheThreadLocal();
    return true;
}

/**
 * check arg is at::Tensor ?
 */
template<typename T>
struct is_at_tensor : std::false_type {};

template<>
struct is_at_tensor<at::Tensor> : std::true_type {};

/**
 * check arg is at::TensorList ?
 */
template<typename T>
struct is_at_tensor_list : std::false_type {};

template<>
struct is_at_tensor_list<at::TensorList> : std::true_type {};

/**
 * find first at::Tensor
 */
template <std::size_t I = 0, typename...Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...>& t, at::Tensor& res) {}

template <std::size_t I = 0, typename... Ts>
typename std::enable_if < I<sizeof...(Ts), void>::type GetFirstTensor(const std::tuple<Ts...> &t, at::Tensor &res)
{
    if constexpr (is_at_tensor<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t);
        return;
    } else if constexpr (is_at_tensor_list<typename std::tuple_element<I, std::tuple<Ts...>>::type>::value) {
        res = std::get<I>(t)[0];
        return;
    }
    return GetFirstTensor<I + 1, Ts...>(t, res);
}

/**
 * get the device
 */
template <typename... Ts>
auto DecodeDevice(Ts&... args) -> at::Device
{
    auto tp = std::make_tuple(args...);
    at::Tensor ft;
    GetFirstTensor(tp, ft);
    return ft.device();
}

/**
 * 异步调用npu执行, 无返回值.
 */
#define EXEC_NPU_CMD_V1(aclnn_api, ...)                                                                                \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    " not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_CMD", "1", __VA_ARGS__);                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(acl_stream);                                                          \
        }                                                                                                              \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        if (hit_cache(acl_stream, #aclnn_api, opApiFuncAddr, __VA_ARGS__)) {                                           \
            break;                                                                                                     \
        }                                                                                                              \
        at_npu::native::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                               \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);                  \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {              \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(acl_stream);                                                      \
            }                                                                                                          \
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

#define EXEC_NPU_CMD_V2(aclnn_api, ...)                                                                                \
    do {                                                                                                               \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                  \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                                \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",               \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    " not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_CMD", "2", __VA_ARGS__);                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(acl_stream);                                                          \
        }                                                                                                              \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        auto deterministic_status = at::globalContext().deterministicAlgorithms();                                     \
        uint32_t aic_num = UINT32_MAX;                                                                                  \
        uint32_t aiv_num = UINT32_MAX;                                                                                  \
        if (c10_npu::is_core_control_enabled()) {                                                            \
            aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);                           \
            aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);                         \
        }                                                                                                              \
        auto acl_call = [copied_params, acl_stream, deterministic_status, aic_num, aiv_num]()->int {                     \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(acl_stream);                                                      \
            }                                                                                                          \
            uint64_t workspace_size = 0;                                                                               \
            uint64_t *workspace_size_addr = &workspace_size;                                                           \
            aclOpExecutor *executor = nullptr;                                                                         \
            aclOpExecutor **executor_addr = &executor;                                                                 \
            InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
            UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);        \
            int api_ret = 0;                                                                                           \
            if (hit_cache_v2(                                                                                          \
               acl_stream, #aclnn_api, opApiFuncAddr, copied_params, &api_ret, deterministic_status, aic_num, aiv_num))  \
            {                                                                                                          \
                return api_ret;                                                                                        \
            }                                                                                                          \
            at_npu::native::SetDeterministicOps(deterministic_status);                                                 \
            if (initMemFunc) {                                                                                         \
                initMemFunc(nullptr, false);                                                                           \
            }                                                                                                          \
            auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                 \
            auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);                \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                      \
            NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                           \
            void *workspace_addr = nullptr;                                                                            \
            at::Tensor workspace_tensor;                                                                               \
            if (workspace_size != 0) {                                                                                 \
                workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);  \
                workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                \
            }                                                                                                          \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                                 \
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

#define EXEC_NPU_CMD(aclnn_api, ...)                                                                                   \
    do {                                                                                                               \
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();                   \
        if (task_queue_enable == 2) {                                                                                  \
            EXEC_NPU_CMD_V2(aclnn_api, __VA_ARGS__);                                                                   \
        } else {                                                                                                       \
            EXEC_NPU_CMD_V1(aclnn_api, __VA_ARGS__);                                                                   \
        }                                                                                                              \
    } while (false)

#define EXEC_NPU_NO_FORMAT_CHECK_CMD_V1(aclnn_api, ...)                                                                \
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
                    " not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_NO_FORMAT_CHECK_CMD", "1", __VA_ARGS__);                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(acl_stream);                                                          \
        }                                                                                                              \
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
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                               \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);                  \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {              \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(acl_stream);                                                      \
            }                                                                                                          \
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

#define EXEC_NPU_NO_FORMAT_CHECK_CMD_V2(aclnn_api, ...)                                                                \
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
                    " not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        OP_EXEC_LOG_WITH_TASK_QUEUE(#aclnn_api, "EXEC_NPU_NO_FORMAT_CHECK_CMD", "2", __VA_ARGS__);                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(acl_stream);                                                          \
        }                                                                                                              \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        auto acl_call = [copied_params, acl_stream]()->int {                                                           \
            if (c10_npu::check_dequeue_need_use(acl_stream)) {                                                         \
                c10_npu::UseStreamResInCurrentThread(acl_stream);                                                      \
            }                                                                                                          \
            uint64_t workspace_size = 0;                                                                               \
            uint64_t *workspace_size_addr = &workspace_size;                                                           \
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
            void *workspace_addr = nullptr;                                                                            \
            at::Tensor workspace_tensor;                                                                               \
            if (workspace_size != 0) {                                                                                 \
                workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);  \
                workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                \
            }                                                                                                          \
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

#define EXEC_NPU_NO_FORMAT_CHECK_CMD(aclnn_api, ...)                                                                   \
    do {                                                                                                               \
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();                   \
        if (task_queue_enable == 2) {                                                                                  \
            EXEC_NPU_NO_FORMAT_CHECK_CMD_V2(aclnn_api, __VA_ARGS__);                                                   \
        } else {                                                                                                       \
            EXEC_NPU_NO_FORMAT_CHECK_CMD_V1(aclnn_api, __VA_ARGS__);                                                   \
        }                                                                                                              \
    } while (false)

#define DO_MATMUL_COMPATIBILITY(aclnn_nz_api, aclnn_nd_api, input1, input2, aclop_func_call)                           \
    do {                                                                                                               \
        if (op_plugin::utils::is_two_tensor_base_format(input1, input2)) {                                             \
            DO_COMPATIBILITY(aclnn_nd_api, aclop_func_call);                                                           \
        } else {                                                                                                       \
            static bool is_support_soc = (c10_npu::GetSocVersion() >= c10_npu::SocVersion::Ascend910B1 &&              \
                                             c10_npu::GetSocVersion() < c10_npu::SocVersion::Ascend310B1) ||           \
                                         (c10_npu::GetSocVersion() > c10_npu::SocVersion::Ascend310B4);                \
            bool is_nz_dtype_valid = (c10_npu::IsAclnnOnly() || ((input1).scalar_type() != at::ScalarType::Float &&    \
                                        (input2).scalar_type() != at::ScalarType::Float));                             \
            if (op_plugin::utils::is_nd_nz_format(input1, input2) && is_support_soc && is_nz_dtype_valid) {            \
                DO_COMPATIBILITY(aclnn_nz_api, aclop_func_call);                                                       \
            } else {                                                                                                   \
                if (!c10_npu::IsAclnnOnly()) {                                                                         \
                    return aclop_func_call;                                                                            \
                }                                                                                                      \
                const torch_npu::NPUStorageDesc &tensor_desc1 =                                                        \
                    torch_npu::NPUBridge::GetNpuStorageImpl(input1)->npu_desc_;                                        \
                const torch_npu::NPUStorageDesc &tensor_desc2 =                                                        \
                    torch_npu::NPUBridge::GetNpuStorageImpl(input2)->npu_desc_;                                        \
                TORCH_CHECK(false,                                                                                     \
                    "matmul got not support format in current device: ",                                               \
                    "(",                                                                                               \
                    tensor_desc1.npu_format_,                                                                          \
                    ", ",                                                                                              \
                    tensor_desc2.npu_format_,                                                                          \
                    ")",                                                                                               \
                    OPS_ERROR(ErrCode::PARAM));                                                                        \
            }                                                                                                          \
        }                                                                                                              \
    } while (0)

template <typename Tuple> class ConvertedParams {
public:
    explicit ConvertedParams(Tuple &&convertedParams, ReleaseHugeMem releaseMemFunc,
                             UnInitHugeMemThreadLocal unInitMemFunc) : convertedParams_(std::move(convertedParams)),
                                                                       releaseMemFunc_(releaseMemFunc),
                                                                       unInitMemFunc_(unInitMemFunc){};
    ConvertedParams(ConvertedParams &&other) : convertedParams_(std::move(other.convertedParams_))
    {
        other.validParams_ = false;
    };
    ConvertedParams &operator=(ConvertedParams &&other)
    {
        if (this == &other) {
            return *this;
        }

        convertedParams_ = std::move(other.convertedParams_);
        validParams_ = true;
        other.validParams_ = false;
        return *this;
    }

    ConvertedParams() = delete;
    ConvertedParams(const ConvertedParams &other) = delete;
    ConvertedParams &operator=(const ConvertedParams &other) = delete;

    ~ConvertedParams()
    {
        if (validParams_) {
            ReleaseConvertTypes(convertedParams_);
            if (releaseMemFunc_) {
                releaseMemFunc_(nullptr, false);
            }
            if (unInitMemFunc_) {
                unInitMemFunc_(nullptr, false);
            }
        }
    }

    const Tuple &GetConvertedParams() const
    {
        return convertedParams_;
    }

    template <size_t i> auto Get()
    {
        return std::get<i>(convertedParams_);
    }

private:
    Tuple convertedParams_;
    ReleaseHugeMem releaseMemFunc_ = nullptr;
    UnInitHugeMemThreadLocal unInitMemFunc_ = nullptr;
    bool validParams_{true};
};

/**
 * 同步调用npu执行，返回把aten的tensor, scalar, array等转换后的参数,
 */
#define EXEC_NPU_CMD_SYNC(aclnn_api, ...)                                                                              \
    [](const char *apiName, const char *workspaceSizeApiName, auto &...args)->auto {                                   \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspaceSizeApiName);                           \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(apiName);                                                   \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                    \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                                \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                         \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");                   \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                       \
        static const auto setPTACacheHashKeyAddr = GetOpApiFuncAddr("SetPTACacheHashKey");                             \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " and ",              \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),          \
                    " not found.", OPS_ERROR(ErrCode::PTR));                                                            \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                                \
        if (c10_npu::check_enqueue_need_use(acl_stream)) {                                                             \
            c10_npu::UseStreamResInCurrentThread(acl_stream);                                                          \
        }                                                                                                              \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                    \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
        ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                              \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                          \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                    \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                          \
        SetPTACacheHashKey setPTACacheHashKeyFunc = reinterpret_cast<SetPTACacheHashKey>(setPTACacheHashKeyAddr);      \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                        \
            initPTACacheThreadLocalFunc();                                                                             \
            setPTAHashKeyFunc(0);                                                                                      \
            if (setPTACacheHashKeyFunc) {                                                                              \
                setPTACacheHashKeyFunc(nullptr, 0);                                                                    \
            }                                                                                                          \
        }                                                                                                              \
        at_npu::native::SetDeterministic();                                                                            \
        if (initMemFunc) {                                                                                             \
            initMemFunc(nullptr, false);                                                                               \
        }                                                                                                              \
        auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);                             \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);             \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        NPU_CHECK_ERROR(workspace_status, "call " #aclnn_api " failed");                                               \
        void *workspace_addr = nullptr;                                                                                \
        at::Tensor workspace_tensor;                                                                                   \
        if (workspace_size != 0) {                                                                                     \
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);                  \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                    \
        }                                                                                                              \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor, apiName]()->int {     \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                          \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                            \
            NPU_CHECK_ERROR(api_ret, "call " #aclnn_api " failed");                                                    \
            return api_ret;                                                                                            \
        };                                                                                                             \
        at_npu::native::OpCommand::RunOpApiV2(apiName, acl_call, true);                                                \
        UnInitCacheThreadLocal();                                                                                      \
        return ConvertedParams<decltype(converted_params)>(std::move(converted_params),                                \
                                                           releaseMemFunc, unInitMemFunc);                             \
    }(#aclnn_api, #aclnn_api "GetWorkspaceSize", __VA_ARGS__)

inline TensorWrapper make_wrapper(const at::Tensor& tensor, c10::optional<int64_t> tensor_dtype)
{
    if (!tensor.defined()) {
        return {tensor, ACL_DT_UNDEFINED};
    }
    if (tensor_dtype.has_value()) {
        aclDataType tensor_acltype = c10_npu::GetAclDataType(tensor_dtype.value());
        int acl_item_size = at_npu::native::OpPreparation::GetAclDataTypeItemSize(tensor_acltype);
        TORCH_CHECK(tensor.itemsize() == acl_item_size,
            "Tensor dtype:", tensor.dtype(), " itemsize:", tensor.itemsize(),
            ", is not compatible with tensor_dtype:", c10_npu::CustomDataTypeToString(tensor_dtype.value()),
            " itemsize:", acl_item_size, OPS_ERROR(ErrCode::PARAM));
        return {tensor, tensor_acltype};
    }

    return {tensor, at_npu::native::OpPreparation::convert_to_acl_data_type(tensor.scalar_type())};
}

inline TensorWrapper make_wrapper(const c10::optional<at::Tensor> &opt_tensor, c10::optional<int64_t> tensor_dtype)
{
    return make_wrapper(opt_tensor.value_or(at::Tensor()), tensor_dtype);
}

inline TensorListWrapper make_wrapper(const at::TensorList& tensorlist, c10::optional<int64_t> tensor_dtype)
{
    if (tensorlist.size() == 0) {
        return {tensorlist, ACL_DT_UNDEFINED};
    }
    if (tensor_dtype.has_value()) {
        aclDataType tensor_acltype = c10_npu::GetAclDataType(tensor_dtype.value());
        int acl_item_size = at_npu::native::OpPreparation::GetAclDataTypeItemSize(tensor_acltype);
        TORCH_CHECK(tensorlist[0].itemsize() == acl_item_size,
            "Tensor dtype:", tensorlist[0].dtype(), " itemsize:", tensorlist[0].itemsize(),
            ", is not compatible with tensor_dtype:", c10_npu::CustomDataTypeToString(tensor_dtype.value()),
            " itemsize:", acl_item_size, OPS_ERROR(ErrCode::PARAM));
        return {tensorlist, tensor_acltype};
    }

    return {tensorlist, at_npu::native::OpPreparation::convert_to_acl_data_type(tensorlist[0].scalar_type())};
}

inline TensorListWrapper make_wrapper(const c10::optional<at::TensorList> &opt_tensorlist, c10::optional<int64_t> tensor_dtype)
{
    return make_wrapper(opt_tensorlist.value_or(at::TensorList()), tensor_dtype);
}
#endif //  TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
