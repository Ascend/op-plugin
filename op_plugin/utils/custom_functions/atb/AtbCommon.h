// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#ifndef OPPLUGIN_UTILS_ATB_COMMON_H
#define OPPLUGIN_UTILS_ATB_COMMON_H
#include <dlfcn.h>
#include <torch/library.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/DeviceUtils.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"
#include "op_plugin/utils/custom_functions/atb/OperationCreate.h"
#include "Utils.h"

namespace atb {

using aclTensor = struct aclTensor;
constexpr int64_t MAX_DIM_NUM = 5;

using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                      const int64_t* stride, int64_t offset, aclFormat format,
                                      const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);
using _aclDestroyTensor = int (*)(const aclTensor *);

using AtbApiFunc = int (*)(void *, uint64_t, atb::Operation *, atb::Context*);

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetApiFuncAddr(#apiName))

inline const char *GetAtbApiLibName(void)
{
    return "libatb.so";
}

inline const char *GetOpApiLibName(void)
{
    return "libopapi.so";
}

inline void *GetApiLibHandler(const char *libName)
{
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void *GetApiFuncAddrInLib(void *handler, const char *libName, const char *apiName)
{
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void *GetApiFuncAddr(const char *apiName)
{
    static auto atbApiHandler = GetApiLibHandler(GetAtbApiLibName());
    if (atbApiHandler != nullptr) {
        auto funcAddr = GetApiFuncAddrInLib(atbApiHandler, GetAtbApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    static auto opApiHandler = GetApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
        if (funcAddr == nullptr) {
            TORCH_CHECK(false, "GetApiFuncAddr not found ", apiName);
        }
    }
}

inline aclTensor *ConvertType(const at::Tensor &tensor)
{
    static const auto aclCreateTensor = reinterpret_cast<_aclCreateTensor>(GetApiFuncAddr("aclCreateTensor"));
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!tensor.defined()) {
        return nullptr;
    }

    at::Tensor at_tensor = tensor;
    if (torch_npu::utils::is_npu(at_tensor)) {
        at_tensor = atb::utils::FormatTrans(tensor);
    }
    if (!at_tensor.is_contiguous()) {
        at_tensor = at_tensor.contiguous();
    }
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = atb::utils::ConvertToAclDataType(scalar_data_type);
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    // if acl_data_type is ACL_STRING, storageDims is empty.
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.");
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    auto acl_tensor =
        aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}

struct TensorStruct {
    void *data_ptr = nullptr;       // at_tensor.storage().data()
    at::ScalarType scalar_type;     // at_tensor.scalar_type()
    size_t nbytes;                  // at_tensor.storage().nbytes()
    size_t itemsize;                // at_tensor.itemsize()
    int64_t storage_offset;         // at_tensor.storage_offset()
    std::vector<int64_t> sizes;     // at_tensor.sizes()
    std::vector<int64_t> strides;   // at_tensor.strides()
    int64_t format;                 // at_tensor format

    TensorStruct(
        void *data_ptr_, at::ScalarType scalar_type_,
        size_t nbytes_, size_t itemsize_, int64_t storage_offset_,
        at::IntArrayRef sizes_, at::IntArrayRef strides_, int64_t format_
    ) : data_ptr(data_ptr_), scalar_type(scalar_type_),
        nbytes(nbytes_), itemsize(itemsize_), storage_offset(storage_offset_),
        sizes(sizes_.vec()), strides(strides_.vec()), format(format_)
    {
    }
};
using TensorStructPtr = std::shared_ptr<TensorStruct>;

inline TensorStructPtr CopyTypeV2(const at::Tensor &tensor)
{
    if (!tensor.defined()) {
        return nullptr;
    }
    at::Tensor at_tensor = tensor;
    if (torch_npu::utils::is_npu(at_tensor)) {
        at_tensor = atb::utils::FormatTrans(tensor);
    }
    if (!at_tensor.is_contiguous()) {
        at_tensor = at_tensor.contiguous();
    }
    int64_t format = 2;
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        at_tensor.scalar_type(),
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        format);
}

template <typename T> T CopyTypeV2(T value)
{
    return value;
}

inline aclTensor *ConvertTypeV2(TensorStructPtr at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (at_tensor == nullptr) {
        return nullptr;
    }
    at::ScalarType scalar_data_type = (*at_tensor).scalar_type;
    aclDataType acl_data_type = atb::utils::ConvertToAclDataType(scalar_data_type);
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    // if acl_data_type is ACL_STRING, storageDims is empty.
    if (acl_data_type != ACL_STRING) {
        TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.");
        storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
    }

    const auto dimNum = (*at_tensor).sizes.size();
    aclFormat format = static_cast<aclFormat>((*at_tensor).format);

    auto acl_tensor = aclCreateTensor(
        (*at_tensor).sizes.data(), (*at_tensor).sizes.size(), acl_data_type, (*at_tensor).strides.data(),
        (*at_tensor).storage_offset, format, storageDims.data(), storageDims.size(), (*at_tensor).data_ptr);
    return acl_tensor;
}

template <typename T> T ConvertTypeV2(T value)
{
    return value;
}

template <typename Tuple, std::size_t... I> auto convert_types_impl_v2(const Tuple &t, std::index_sequence<I...>)
{
    return std::make_tuple(ConvertTypeV2(std::get<I>(t))...);
}

template <typename... Ts> constexpr auto ConvertTypesV2(
    const std::tuple<Ts...> &args,
    uint64_t *workspace_size_addr, atb::Operation **op_addr, atb::Context *contextPtr)
{
    auto convert_args = convert_types_impl_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
    auto appends = std::make_tuple(workspace_size_addr, op_addr, contextPtr);
    return std::tuple_cat(convert_args, appends);
}

template <typename... Ts> constexpr auto CopyTypesV2(Ts &...args)
{
    return std::make_tuple(CopyTypeV2(args)...);
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
    using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple> auto ConvertToOpApiFunc(const Tuple &params, void *opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(atb::Context *context) {}

inline void Release(aclTensor *p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
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

#define EXEC_ATB_CMD(atb_api, ...)                                                                                \
    do {                                                                                                          \
        static const auto getWorkspaceSizeFuncAddr = GetApiFuncAddr(#atb_api "GetWorkspaceSize");                 \
        static const auto AtbApiFuncAddr = GetApiFuncAddr(#atb_api);                                              \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && AtbApiFuncAddr != nullptr, #atb_api, " or ",           \
                    #atb_api "GetWorkspaceSize", " not in ", GetAtbApiLibName(), ", or ", GetAtbApiLibName(),     \
                    "not found.");                                                                                \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                           \
        auto copied_params = CopyTypesV2(__VA_ARGS__);                                                            \
        auto hash_id = computeHash(std::string(#atb_api), __VA_ARGS__);                                           \
        auto atb_call = [copied_params, acl_stream, hash_id]()->int {                                             \
            auto contextPtr = GetContext(acl_stream);                                                             \
            uint64_t workspace_size = 0;                                                                          \
            uint64_t *workspace_size_addr = &workspace_size;                                                      \
            OpParamCache<uint64_t>& opParamCache = OpParamCache<uint64_t>::getInstance();                         \
            atb::Operation *op = opParamCache.getOperation(hash_id);                                              \
            atb::Operation **op_addr = &op;                                                                       \
            int api_ret = 0;                                                                                      \
            auto converted_params = ConvertTypesV2(copied_params, workspace_size_addr, op_addr, contextPtr);      \
            auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);           \
            auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                 \
            opParamCache.saveOperation(hash_id, op);                                                              \
            TORCH_CHECK(workspace_status == 0, "call " #atb_api"GetWorkspaceSize failed");                        \
            void *workspace_addr = nullptr;                                                                       \
            at::Tensor workspace_tensor;                                                                          \
            if (workspace_size != 0) {                                                                            \
                workspace_tensor = at_npu::native::allocate_workspace(workspace_size, acl_stream);                \
                workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                           \
            }                                                                                                     \
            AtbApiFunc atbApiFunc = reinterpret_cast<AtbApiFunc>(AtbApiFuncAddr);                                 \
            api_ret = atbApiFunc(workspace_addr, workspace_size, op, contextPtr);                                 \
            TORCH_CHECK(api_ret == 0, "call " #atb_api " failed");                                                \
            ReleaseConvertTypes(converted_params);                                                                \
            return api_ret;                                                                                       \
        };                                                                                                        \
        at_npu::native::OpCommand::RunOpApi(#atb_api, atb_call);                                                  \
    } while (false)


atb::Tensor AtTensor2AtbTensor(const at::Tensor atTensor);
atb::Context* GetContext(aclrtStream stream);
at::Tensor GetWorkspaceTensor(uint64_t workspaceSize, aclrtStream stream);
uint64_t OperationSetup(atb::VariantPack variantPack, atb::Operation *operation, atb::Context* contextPtr);
class ParamSetter {
public:
    ParamSetter& Input(const at::Tensor &tensor);
    ParamSetter& Input(const c10::optional<at::Tensor> &tensor);
    ParamSetter& Output(at::Tensor &tensor);
    atb::VariantPack variantPack;
};

class ContextManager {
public:
    static ContextManager& GetInstance();
    atb::Context* GetContext(aclrtStream stream);
    ~ContextManager();

    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;

private:
    ContextManager();
    std::once_flag createFlag;
    atb::Context* atbContext;
};

void RunAtbCmd(atb::Operation *op, const ParamSetter &paramsetter, const std::string &name);

} // namespace atb

#endif
