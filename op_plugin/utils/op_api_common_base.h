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

#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_BASE_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_BASE_H_

#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>
#include <vector>
#include <filesystem>
#include <functional>
#include <type_traits>
#include <ATen/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <acl/acl_base.h>
#include "op_plugin/utils/KernelNpuOutputSize.h"
#include "op_plugin/utils/KernelNpuOutputDtype.h"
#include "op_plugin/utils/KernelNpuNewParams.h"
#include "op_plugin/utils/OpUtils.h"
#include "op_plugin/utils/op_log.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/flopcount/FlopCount.h"
#include "torch_npu/csrc/flopcount/FlopCounter.h"
#include "torch_npu/csrc/custom_dtype/Init.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclScalarList aclScalarList;

typedef aclTensor *(*_aclCreateTensor)(const int64_t *view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t *stride, int64_t offset, aclFormat format,
                                       const int64_t *storage_dims, uint64_t storage_dims_num, void *tensor_data);
typedef aclScalar *(*_aclCreateScalar)(void *value, aclDataType data_type);
typedef aclIntArray *(*_aclCreateIntArray)(const int64_t *value, uint64_t size);
typedef aclFloatArray *(*_aclCreateFloatArray)(const float *value, uint64_t size);
typedef aclBoolArray *(*_aclCreateBoolArray)(const bool *value, uint64_t size);
typedef aclTensorList *(*_aclCreateTensorList)(const aclTensor *const *value, uint64_t size);
typedef aclScalarList *(*_aclCreateScalarList)(const aclScalar *const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor *tensor);
typedef int (*_aclDestroyScalar)(const aclScalar *scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray *array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray *array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray *array);
typedef int (*_aclDestroyTensorList)(const aclTensorList *array);
typedef int (*_aclDestroyScalarList)(const aclScalarList *array);

using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);

TORCH_NPU_API void *GetOpApiFuncAddr(const char *apiName); // GET_OP_API_FUNC
#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetOpApiFuncAddr(#apiName))

TORCH_NPU_API aclTensor *ConvertType(const at::Tensor &at_tensor);
TORCH_NPU_API aclScalar *ConvertType(const at::Scalar &at_scalar);
TORCH_NPU_API aclIntArray *ConvertType(const at::IntArrayRef &at_array);
TORCH_NPU_API aclIntArray *ConvertType(const at::ArrayRef<c10::SymInt> &at_array);
TORCH_NPU_API aclBoolArray *ConvertType(const at::ArrayRef<bool> &value);
TORCH_NPU_API aclTensorList *ConvertType(const at::TensorList &at_tensor_list);
TORCH_NPU_API aclScalarList *ConvertType(const at::ArrayRef<at::Scalar> &at_scalar_list);
TORCH_NPU_API aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor);
TORCH_NPU_API aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array);
TORCH_NPU_API aclIntArray *ConvertType(const c10::OptionalArrayRef<c10::SymInt> &opt_array);
TORCH_NPU_API aclIntArray *ConvertType(const c10::OptionalIntArrayRef &opt_array);
TORCH_NPU_API aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar);
TORCH_NPU_API aclDataType ConvertType(const at::ScalarType scalarType);
TORCH_NPU_API aclTensor *ConvertType(const TensorWrapper &tensor_r);
TORCH_NPU_API aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper);

template <std::size_t N> inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

template <typename T> T ConvertType(T value)
{
    return value;
}

struct TensorStruct {
    void *data_ptr = nullptr;       // at_tensor.storage().data()
    aclDataType acl_type;           // aclDataType of at_tensor
    aclFormat acl_format;
    size_t nbytes;                  // at_tensor.storage().nbytes()
    size_t itemsize;                // at_tensor.itemsize()
    int64_t storage_offset;         // at_tensor.storage_offset()
    std::vector<int64_t> sizes;     // at_tensor.sizes()
    std::vector<int64_t> strides;   // at_tensor.strides()
    std::vector<int64_t> storage_sizes;

    TensorStruct(
        void *data_ptr_, aclDataType acl_type_, aclFormat acl_format_,
        size_t nbytes_, size_t itemsize_, int64_t storage_offset_,
        at::IntArrayRef sizes_, at::IntArrayRef strides_, at::IntArrayRef storage_sizes_
    ) : data_ptr(data_ptr_), acl_type(acl_type_), acl_format(acl_format_),
        nbytes(nbytes_), itemsize(itemsize_), storage_offset(storage_offset_),
        sizes(sizes_.vec()), strides(strides_.vec()), storage_sizes(storage_sizes_.vec())
    {
    }
};
using TensorStructPtr = std::shared_ptr<TensorStruct>;

TORCH_NPU_API aclTensor *ConvertTypeV2(TensorStructPtr at_tensor);
TORCH_NPU_API aclScalar *ConvertTypeV2(const at::Scalar &at_scalar);
TORCH_NPU_API aclIntArray *ConvertTypeV2(const std::vector<int64_t> &int_list);
TORCH_NPU_API aclBoolArray *ConvertTypeV2(const std::vector<bool> &value);
TORCH_NPU_API aclTensorList *ConvertTypeV2(const std::vector<TensorStructPtr> &at_tensor_list);
TORCH_NPU_API aclScalarList *ConvertTypeV2(const std::vector<at::Scalar> &at_scalar_list);
TORCH_NPU_API aclIntArray *ConvertTypeV2(const c10::optional<std::vector<int64_t>> &opt_array);
TORCH_NPU_API aclScalar *ConvertTypeV2(const c10::optional<at::Scalar> &opt_scalar);
TORCH_NPU_API aclDataType ConvertTypeV2(const at::ScalarType scalarType);
TORCH_NPU_API char* ConvertTypeV2(const std::string &str);

template <std::size_t N> inline aclBoolArray *ConvertTypeV2(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

template <typename T> T ConvertTypeV2(T value)
{
    return value;
}

TORCH_NPU_API TensorStructPtr CopyTypeV2(const at::Tensor &at_tensor);
TORCH_NPU_API TensorStructPtr CopyTypeV2(const TensorWrapper &tensor_r);
TORCH_NPU_API std::vector<int64_t> CopyTypeV2(const at::IntArrayRef &at_array);
TORCH_NPU_API std::vector<int64_t> CopyTypeV2(const at::ArrayRef<c10::SymInt> &at_array);
TORCH_NPU_API std::vector<bool> CopyTypeV2(const at::ArrayRef<bool> &value);
TORCH_NPU_API std::vector<TensorStructPtr> CopyTypeV2(const at::TensorList &at_tensor_list);
TORCH_NPU_API std::vector<TensorStructPtr> CopyTypeV2(const TensorListWrapper &tensor_list_wrapper);
TORCH_NPU_API std::vector<at::Scalar> CopyTypeV2(const at::ArrayRef<at::Scalar> &at_scalar_list);
TORCH_NPU_API TensorStructPtr CopyTypeV2(const c10::optional<at::Tensor> &opt_tensor);
TORCH_NPU_API c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::optional<at::IntArrayRef> &opt_array);
TORCH_NPU_API c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalArrayRef<c10::SymInt> &opt_array);
TORCH_NPU_API c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalIntArrayRef &opt_array);
TORCH_NPU_API std::string CopyTypeV2(char* str);

template <std::size_t N> inline std::array<bool, N> CopyTypeV2(const std::array<bool, N> &value)
{
    return value;
}

template <typename T> T CopyTypeV2(T value)
{
    return value;
}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar *p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray *p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclBoolArray *p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList *p)
{
    static const auto aclDestroyTensorList = GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

inline void Release(aclScalarList *p)
{
    static const auto aclDestroyScalarList = GET_OP_API_FUNC(aclDestroyScalarList);
    if (aclDestroyScalarList == nullptr) {
        return;
    }

    aclDestroyScalarList(p);
}

template <typename T> void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I> void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple> void ReleaseConvertTypes(Tuple &t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Tuple, std::size_t... I> auto convert_types_impl_v2(const Tuple &t, std::index_sequence<I...>)
{
    return std::make_tuple(ConvertTypeV2(std::get<I>(t))...);
}

template <typename... Ts> constexpr auto ConvertTypesV2(
    const std::tuple<Ts...> &args,
    uint64_t *workspace_size_addr, aclOpExecutor **executor_addr)
{
    auto convert_args = convert_types_impl_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
    auto appends = std::make_tuple(workspace_size_addr, executor_addr);
    return std::tuple_cat(convert_args, appends);
}

template <typename... Ts> constexpr auto CopyTypesV2(Ts &...args)
{
    return std::make_tuple(CopyTypeV2(args)...);
}

TORCH_NPU_API void MemcpyToBufImpl(const void* data, size_t size);

#define MEMCPY_TO_BUF(data_expression, size_expression) MemcpyToBufImpl(data_expression, size_expression)

template <std::size_t N> void add_param_to_buf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), static_cast<int64_t>(value.size() * sizeof(bool)));
}

template <typename T> void add_param_to_buf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

TORCH_NPU_API void add_param_to_buf(const at::Tensor &);
TORCH_NPU_API void add_param_to_buf(const at::Scalar &);
TORCH_NPU_API void add_param_to_buf(const at::IntArrayRef &);
TORCH_NPU_API void add_param_to_buf(const at::ArrayRef<c10::SymInt> &);
TORCH_NPU_API void add_param_to_buf(const at::ArrayRef<bool> &);
TORCH_NPU_API void add_param_to_buf(const at::TensorList &);
TORCH_NPU_API void add_param_to_buf(const at::ArrayRef<at::Scalar> &);
TORCH_NPU_API void add_param_to_buf(const c10::optional<at::Tensor> &);
TORCH_NPU_API void add_param_to_buf(const c10::optional<at::IntArrayRef> &);
TORCH_NPU_API void add_param_to_buf(const c10::OptionalArrayRef<c10::SymInt> &);
TORCH_NPU_API void add_param_to_buf(const c10::OptionalIntArrayRef &);
TORCH_NPU_API void add_param_to_buf(const c10::optional<at::Scalar> &);
TORCH_NPU_API void add_param_to_buf(const at::ScalarType);
TORCH_NPU_API void add_param_to_buf(const string &);
TORCH_NPU_API void add_param_to_buf(char *);
TORCH_NPU_API void add_param_to_buf(const char *);
TORCH_NPU_API void add_param_to_buf(const TensorWrapper &tensor_r);
TORCH_NPU_API void add_param_to_buf(const TensorListWrapper &tensor_list_wrapper);
TORCH_NPU_API void add_param_to_buf();

template <typename T, typename... Args> void add_param_to_buf(const T &arg, Args &...args)
{
    add_param_to_buf(arg);
    add_param_to_buf(args...);
}

template <std::size_t N> void add_param_to_buf_v2(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), static_cast<int64_t>(value.size() * sizeof(bool)));
}

template <typename T> void add_param_to_buf_v2(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

TORCH_NPU_API void add_param_to_buf_v2(TensorStructPtr);
TORCH_NPU_API void add_param_to_buf_v2(const at::Scalar &);
TORCH_NPU_API void add_param_to_buf_v2(const std::vector<int64_t> &);
TORCH_NPU_API void add_param_to_buf_v2(const std::vector<bool> &);
TORCH_NPU_API void add_param_to_buf_v2(const std::vector<TensorStructPtr> &);
TORCH_NPU_API void add_param_to_buf_v2(const std::vector<at::Scalar> &);
TORCH_NPU_API void add_param_to_buf_v2(const c10::optional<std::vector<int64_t>> &);
TORCH_NPU_API void add_param_to_buf_v2(const c10::optional<at::Scalar> &);
TORCH_NPU_API void add_param_to_buf_v2(const at::ScalarType);
TORCH_NPU_API void add_param_to_buf_v2(const string &);
TORCH_NPU_API void add_param_to_buf_v2(char *);
TORCH_NPU_API void add_param_to_buf_v2(const char *);
TORCH_NPU_API void add_param_to_buf_v2();

template <typename T, typename... Args> void add_param_to_buf_v2(const T &arg, Args &...args)
{
    add_param_to_buf_v2(arg);
    add_param_to_buf_v2(args...);
}

template <typename ...Ts, std::size_t ...i>
void add_params_to_buf_v2(const std::tuple<Ts...> &t, std::index_sequence<i...>)
{
    (add_param_to_buf_v2(std::get<i>(t)), ...);
}

template <typename Function, typename Tuple, size_t... I> auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple> auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr, std::index_sequence<I...>)
{
    typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple> auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

struct CacheParams {
public:
    bool GetDeterministicStatus() const;
    uint32_t GetAicNum() const;
    uint32_t GetAivNum() const;

private:
    bool deterministic_status_ = false;
    uint32_t aic_num_ = UINT32_MAX;
    uint32_t aiv_num_ = UINT32_MAX;
    friend CacheParams GetCacheParams();
};

TORCH_NPU_API CacheParams GetCacheParams();

TORCH_NPU_API bool CheckAndInitFunc(const char* aclnn_api);
TORCH_NPU_API void AddCacheConfigParams(aclrtStream acl_stream, CacheParams cache_params);
TORCH_NPU_API bool ExecuteCachedOp(aclrtStream acl_stream, const char* aclnn_api, void* phrase2);

TORCH_NPU_API bool CheckAndInitFuncV2(const char* aclnn_api);
TORCH_NPU_API void AddCacheConfigParamsV2(aclrtStream acl_stream, CacheParams cache_params, const char* aclnn_api);
TORCH_NPU_API bool ExecuteCachedOpV2(aclrtStream acl_stream, const char* aclnn_api, void* phrase2, int* api_ret);

template <typename... Args>
void BuildCacheHashParams(aclrtStream acl_stream, const char* aclnn_api, Args&&... args)
{
    CacheParams cache_params = GetCacheParams();
    AddCacheConfigParams(acl_stream, cache_params);
    add_param_to_buf(std::string(aclnn_api), args...);
}

template <typename... Ts>
void BuildCacheHashParamsV2(aclrtStream acl_stream, CacheParams cache_params, const char* aclnn_api, const std::tuple<Ts...> &args)
{
    AddCacheConfigParamsV2(acl_stream, cache_params, aclnn_api);
    add_params_to_buf_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename... Args>
bool hit_cache_ext(aclrtStream acl_stream, const char* aclnn_api, void* phrase2, Args&&... args)
{
    // 步骤1：检查缓存功能是否可用
    if (!CheckAndInitFunc(aclnn_api)) {
        return false;
    }
    // 步骤2：加入缓存
    BuildCacheHashParams(acl_stream, aclnn_api, std::forward<Args>(args)...);
    // 步骤3：执行缓存的算子
    return ExecuteCachedOp(acl_stream, aclnn_api, phrase2);
}


template <typename ...Ts>
bool hit_cache_v2_ext(
    aclrtStream acl_stream, const char *aclnn_api, void *phrase2, const std::tuple<Ts...> &args, int* api_ret,
    CacheParams cache_params)
{
    // 步骤1：检查缓存功能并初始化
    if (!CheckAndInitFuncV2(aclnn_api)) {
        return false;
    }
    // 步骤2：加入缓存
    BuildCacheHashParamsV2(acl_stream, cache_params, aclnn_api, args);
    // 步骤3：执行缓存的算子
    return ExecuteCachedOpV2(acl_stream, aclnn_api, phrase2, api_ret);
}

TORCH_NPU_API void GetApiFunc(
    const char* api_name,
    const char* workspace_api_name,
    void*& opApiFuncAddr,
    void*& getWorkspaceSizeFuncAddr
);
TORCH_NPU_API void InitExecCommonCtx();
TORCH_NPU_API void UnInitExecCommonCtx();
TORCH_NPU_API aclrtStream GetAclStream();
TORCH_NPU_API void SetExecConfig();
TORCH_NPU_API void SetExecConfigV2(CacheParams cache_params);
TORCH_NPU_API int ExecuteApiFunc(
    const void* opApiFuncAddr,
    aclrtStream acl_stream,
    uint64_t workspace_size,
    aclOpExecutor* executor
);

TORCH_NPU_API void InitExecSubTheadCtx(aclrtStream acl_stream);

using PROC_FUNC = std::function<int()>;

TORCH_NPU_API void RunAclCall(const string &op_name, const PROC_FUNC &func);

#define EXEC_NPU_CMD_V1_EXT(aclnn_api, ...)                                                                            \
    do {                                                                                                               \
        /* 1.初始化函数指针 */                                                                                           \
        static void* opApiFuncAddr = nullptr;                                                                          \
        static void* getWorkspaceSizeFuncAddr = nullptr;                                                               \
        GetApiFunc(#aclnn_api, #aclnn_api "GetWorkspaceSize",                                                          \
                                      opApiFuncAddr, getWorkspaceSizeFuncAddr);                                        \
        /* 2.初始化执行上下文：内存管理相关 */                                                                              \
        InitExecCommonCtx();                                                                                          \
        /* 3.获取NPU流 */                                                                                               \
        auto acl_stream = GetAclStream();                                                                              \
        /* 4.缓存查询 */                                                                                                \
        if (hit_cache_ext(acl_stream, #aclnn_api, opApiFuncAddr, __VA_ARGS__)) {                                       \
            break;                                                                                                     \
        }                                                                                                              \
        /* 5.设置执行配置：确定性算法 */                                                                                   \
        SetExecConfig();                                                                                               \
        /* 6.参数转换、获取workspace_size和executor（涉及可变参数未封装） */                                                  \
        uint64_t workspace_size = 0;                                                                                   \
        uint64_t *workspace_size_addr = &workspace_size;                                                               \
        aclOpExecutor *executor = nullptr;                                                                             \
        aclOpExecutor **executor_addr = &executor;                                                                     \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                         \
        auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);                    \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                          \
        /* 7.定义子线程执行函数 */                                                                                        \
        auto acl_call = [converted_params, workspace_size, acl_stream, executor, opApiFuncAddr]()->int {               \
            /* 7.1.初始化子进程上下文 */                                                                                  \
            InitExecSubTheadCtx(acl_stream);                                                                           \
            /* 7.2.执行aclnn函数 */                                                                                     \
            auto api_ret = ExecuteApiFunc(opApiFuncAddr, acl_stream, workspace_size, executor);                        \
            /* 7.3.清理参数、内存 */                                                                                     \
            ReleaseConvertTypes(converted_params);                                                                     \
            UnInitExecCommonCtx();                                                                                     \
            return api_ret;                                                                                            \
        };                                                                                                             \
        RunAclCall(#aclnn_api, acl_call);                                                                              \
    } while (false)

#define EXEC_NPU_CMD_V2_EXT(aclnn_api, ...)                                                                            \
    do {                                                                                                               \
        /* 1.初始化函数指针 */                                                                                           \
        static void* opApiFuncAddr = nullptr;                                                                          \
        static void* getWorkspaceSizeFuncAddr = nullptr;                                                               \
        GetApiFunc(#aclnn_api, #aclnn_api "GetWorkspaceSize",                                                          \
                                      opApiFuncAddr, getWorkspaceSizeFuncAddr);                                        \
        /* 2.获取NPU流 */                                                                                               \
        auto acl_stream = GetAclStream();                                                                              \
        /* 3.拷贝参数列表 */                                                                                             \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                                 \
        /* 4.获取查询缓存所需参数（部分参数需要在主线程获取） */                                                                \
        CacheParams cache_params = GetCacheParams();                                                                   \
        /* 5.定义子线程执行函数 */                                                                                        \
        auto acl_call = [copied_params, acl_stream, cache_params]()->int {                                             \
            /* 5.1.初始化子进程上下文 */                                                                                  \
            InitExecSubTheadCtx(acl_stream);                                                                           \
            int api_ret = 0;                                                                                           \
            /* 5.2.缓存查询 */                                                                                          \
            if (hit_cache_v2_ext(acl_stream, #aclnn_api, opApiFuncAddr, copied_params, &api_ret, cache_params))        \
            {                                                                                                          \
                return api_ret;                                                                                        \
            }                                                                                                          \
            /* 5.3.设置执行配置 */                                                                                       \
            SetExecConfigV2(cache_params);                                                                             \
            /* 5.4.初始化执行上下文：内存管理相关 */                                                                         \
            InitExecCommonCtx();                                                                                       \
            /* 5.5.参数转换、获取workspace_size和executor（涉及可变参数未封装） */                                            \
            uint64_t workspace_size = 0;                                                                               \
            uint64_t *workspace_size_addr = &workspace_size;                                                           \
            aclOpExecutor *executor = nullptr;                                                                         \
            aclOpExecutor **executor_addr = &executor;                                                                 \
            auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, executor_addr);                 \
            auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);                \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                      \
            /* 5.6.执行aclnn函数 */                                                                                     \
            api_ret = ExecuteApiFunc(opApiFuncAddr, acl_stream, workspace_size, executor);                             \
            /* 5.7.清理参数、内存 */                                                                                      \
            ReleaseConvertTypes(converted_params);                                                                     \
            UnInitExecCommonCtx();                                                                                     \
            return api_ret;                                                                                            \
        };                                                                                                             \
        /* 6.调用子线程执行函数 */                                                                                        \
        RunAclCall(#aclnn_api, acl_call);                                                                              \
    } while (false)

#define EXEC_NPU_CMD_EXT(aclnn_api, ...)                                                                               \
    do {                                                                                                               \
        static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();                   \
        if (task_queue_enable == 2) {                                                                                  \
            EXEC_NPU_CMD_V2_EXT(aclnn_api, __VA_ARGS__);                                                               \
        } else {                                                                                                       \
            EXEC_NPU_CMD_V1_EXT(aclnn_api, __VA_ARGS__);                                                               \
        }                                                                                                              \
    } while (false)

#endif //  TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_BASE_H_
