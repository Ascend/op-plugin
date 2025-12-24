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

#include <fstream>
#include <sys/stat.h>
#include <dlfcn.h>
#include <vector>
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

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;
extern const std::vector<std::string> g_custom_lib_path;
extern const std::vector<std::string> g_default_custom_lib_path;

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

#define MEMCPY_TO_BUF(data_expression, size_expression)                                                                \
    if (g_hash_offset + (size_expression) > g_hash_buf_size) {                                                         \
        g_hash_offset = g_hash_buf_max_size;                                                                           \
        return;                                                                                                        \
    }                                                                                                                  \
    memcpy(g_hash_buf + g_hash_offset, data_expression, size_expression);                                              \
    g_hash_offset += size_expression;

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

inline void *GetOpApiFuncAddr(const char *apiName)
{
    if (!g_custom_lib_path.empty()) {
        for (auto &it : g_custom_lib_path) {
            auto cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    // check owner
                    if (!checkOwner(cust_opapi_lib)) {
                        continue;
                    }
                    ASCEND_LOGI("%s is found in %s.", apiName, cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in custom lib.", apiName);
    }

    if (!g_default_custom_lib_path.empty()) {
        for (auto &it : g_default_custom_lib_path) {
            auto default_cust_opapi_lib = real_path(it + "/" + GetCustOpApiLibName());
            if (default_cust_opapi_lib.empty()) {
                continue;
            }
            auto custOpApiHandler = GetOpApiLibHandler(default_cust_opapi_lib.c_str());
            if (custOpApiHandler != nullptr) {
                auto funcAddr =
                    GetOpApiFuncAddrInLib(custOpApiHandler, GetCustOpApiLibName(), apiName);
                if (funcAddr != nullptr) {
                    // check owner
                    if (!checkOwner(default_cust_opapi_lib)) {
                        continue;
                    }
                    ASCEND_LOGI("%s is found in %s.", apiName, default_cust_opapi_lib.c_str());
                    return funcAddr;
                }
            }
        }
        ASCEND_LOGI("%s is not in default custom lib.", apiName);
    }

    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiMathHandler, "libopapi_math.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiNnHandler, "libopapi_nn.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiCvHandler, "libopapi_cv.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiTransformerHandler, "libopapi_transformer.so", apiName);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(opapiLegacyHandler, "libopapi_legacy.so", apiName);

    static auto opApiHandler = GetOpApiLibHandler(GetOpApiLibName());
    if (opApiHandler != nullptr) {
        auto funcAddr = GetOpApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }
    return GetOpApiFuncAddrFromFeatureLib(apiName);
}

inline aclTensor *ConvertType(const at::Tensor &at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
    at::ScalarType scalar_data_type = at_tensor.scalar_type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
        format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    if (at_npu::native::OpPreparation::is_scalar_wrapped_to_tensor(at_tensor)) {
        c10::Scalar expScalar = at_tensor.item();
        at::Tensor aclInput = at_npu::native::OpPreparation::copy_scalar_to_device(expScalar, scalar_data_type);
        return aclCreateTensor(aclInput.sizes().data(), aclInput.sizes().size(), acl_data_type,
                               aclInput.strides().data(), aclInput.storage_offset(), format, storageDims.data(),
                               storageDims.size(), const_cast<void *>(aclInput.storage().data()));
    }

    auto acl_tensor =
        aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_data_type, at_tensor.strides().data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclScalar *ConvertType(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

inline aclIntArray *ConvertType(const at::IntArrayRef &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(at_array.data(), at_array.size());
    return array;
}

inline aclIntArray *ConvertType(const at::ArrayRef<c10::SymInt> &at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    auto array = aclCreateIntArray(int_array.data(), int_array.size());
    return array;
}

template <std::size_t N> inline aclBoolArray *ConvertType(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclBoolArray *ConvertType(const at::ArrayRef<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclTensorList *ConvertType(const at::TensorList &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclScalarList *ConvertType(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertType(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline aclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertType(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertType(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }

    return nullptr;
}

inline aclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }

    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType);
}

inline aclTensor *ConvertType(const TensorWrapper &tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    const at::Tensor &at_tensor = tensor_r.tensor_;

    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));

    aclDataType acl_data_type = tensor_r.dtype;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
        format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                OPS_ERROR(ErrCode::VALUE));
            storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            if (acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4) {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize() * FP4_IN_INT8);
                if (at_tensor.sizes().size() == 1) {
                    wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
                } else if (at_tensor.sizes().size() > 1) {
                    if (wrapperStride[at_tensor.sizes().size() - 1] == 1) {
                        wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] =
                            wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                        wrapperShape[at_tensor.sizes().size() - 1] =
                            wrapperShape[at_tensor.sizes().size() - 1] * FP4_IN_INT8;
                    } else if (wrapperStride[at_tensor.sizes().size() - PENULTIMATE_DIM] == 1) {
                        wrapperStride[at_tensor.sizes().size() - 1] =
                            wrapperStride[at_tensor.sizes().size() - 1] * FP4_IN_INT8;
                        wrapperShape[at_tensor.sizes().size() - PENULTIMATE_DIM] =
                            wrapperShape[at_tensor.sizes().size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                    }
                    
                    for (auto i = 0; i < at_tensor.sizes().size() - PENULTIMATE_DIM; i++) {
                        wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
                    }
                } else {
                    TORCH_CHECK(false, "unsupported tensor wrapper strides in 4-bit dtype.", OPS_ERROR(ErrCode::VALUE));
                }
            } else {
                storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
            }
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
                        at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
                        const_cast<void *>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclTensorList *ConvertType(const TensorListWrapper &tensor_list_wrapper)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
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

inline aclTensor *ConvertTypeV2(TensorStructPtr at_tensor)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) {
        return nullptr;
    }

    if (at_tensor == nullptr) {
        return nullptr;
    }
    aclDataType acl_data_type = (*at_tensor).acl_type;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = op_infer::array_to_small_vector((*at_tensor).strides);
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = op_infer::array_to_small_vector((*at_tensor).sizes);

    const auto dimNum = (*at_tensor).sizes.size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsBaseFormatType((*at_tensor).acl_format)) {
        format = (*at_tensor).acl_format;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            storageDims = (*at_tensor).storage_sizes;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK((*at_tensor).itemsize > 0, "the itemsize of tensor must be greater than 0.",
                        OPS_ERROR(ErrCode::VALUE));
            if (acl_data_type == ACL_FLOAT4_E2M1 || acl_data_type == ACL_FLOAT4_E1M2 || acl_data_type == ACL_INT4) {
                storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize * FP4_IN_INT8);
                if ((*at_tensor).sizes.size() == 1) {
                    wrapperShape[0] = wrapperShape[0] * FP4_IN_INT8;
                } else if ((*at_tensor).sizes.size() > 1 && wrapperStride[(*at_tensor).sizes.size() - 1] == 1) {
                    wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] =
                        wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                    for (auto i = 0; i < (*at_tensor).sizes.size() - PENULTIMATE_DIM; i++) {
                        wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
                    }
                    wrapperShape[(*at_tensor).sizes.size() - 1] =
                        wrapperShape[(*at_tensor).sizes.size() - 1] * FP4_IN_INT8;
                } else if ((*at_tensor).sizes.size() > 1 &&
                           wrapperStride[(*at_tensor).sizes.size() - PENULTIMATE_DIM] == 1) {
                    wrapperStride[(*at_tensor).sizes.size() - 1] =
                        wrapperStride[(*at_tensor).sizes.size() - 1] * FP4_IN_INT8;
                    for (auto i = 0; i < (*at_tensor).sizes.size() - PENULTIMATE_DIM; i++) {
                        wrapperStride[i] = wrapperStride[i] * FP4_IN_INT8;
                    }
                    wrapperShape[(*at_tensor).sizes.size() - PENULTIMATE_DIM] =
                        wrapperShape[(*at_tensor).sizes.size() - PENULTIMATE_DIM] * FP4_IN_INT8;
                } else {
                    TORCH_CHECK(false, "unsupported tensor warrper strides in 4-bit dtype.", OPS_ERROR(ErrCode::VALUE));
                }
            } else {
                storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
            }
        }
    }

    auto acl_tensor = aclCreateTensor(
        wrapperShape.data(), (*at_tensor).sizes.size(), acl_data_type, wrapperStride.data(),
        (*at_tensor).storage_offset, format, storageDims.data(), storageDims.size(), (*at_tensor).data_ptr);
    return acl_tensor;
}

inline TensorStructPtr CopyTypeV2(const at::Tensor &at_tensor)
{
    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(at_tensor.scalar_type());
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        acl_data_type,
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_);
}

inline TensorStructPtr CopyTypeV2(const TensorWrapper &tensor_r)
{
    const at::Tensor &at_tensor = tensor_r.tensor_;
    if (!at_tensor.defined()) {
        return nullptr;
    }
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));
    return std::make_shared<TensorStruct>(
        const_cast<void *>(at_tensor.storage().data()),
        tensor_r.dtype,
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_,
        at_tensor.storage().nbytes(),
        at_tensor.itemsize(),
        at_tensor.storage_offset(),
        at_tensor.sizes(),
        at_tensor.strides(),
        torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_);
}

inline aclScalar *ConvertTypeV2(const at::Scalar &at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type = at_npu::native::OpPreparation::convert_to_acl_data_type(scalar_data_type);
    aclScalar *acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double:
            {
                double value = at_scalar.toDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Long:
            {
                int64_t value = at_scalar.toLong();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::Bool:
            {
                bool value = at_scalar.toBool();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        case at::ScalarType::ComplexDouble:
            {
                auto value = at_scalar.toComplexDouble();
                acl_scalar = aclCreateScalar(&value, acl_data_type);
                break;
            }
        default:
            acl_scalar = nullptr;
            break;
    }

    return acl_scalar;
}

inline aclIntArray *ConvertTypeV2(const std::vector<int64_t> &int_list)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto array = aclCreateIntArray(int_list.data(), int_list.size());
    return array;
}

inline std::vector<int64_t> CopyTypeV2(const at::IntArrayRef &at_array)
{
    return at_array.vec();
}

inline std::vector<int64_t> CopyTypeV2(const at::ArrayRef<c10::SymInt> &at_array)
{
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    return int_array.vec();
}

template <std::size_t N> inline aclBoolArray *ConvertTypeV2(const std::array<bool, N> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

template <std::size_t N> inline std::array<bool, N> CopyTypeV2(const std::array<bool, N> &value)
{
    return value;
}

inline aclBoolArray *ConvertTypeV2(const std::vector<bool> &value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    bool *value_ptr = reinterpret_cast<bool *>(malloc(value.size() * sizeof(bool)));
    for (size_t i = 0; i < value.size(); i++) {
        value_ptr[i] = value[i];
    }
    auto array = aclCreateBoolArray(value_ptr, value.size());
    free(value_ptr);
    return array;
}

inline std::vector<bool> CopyTypeV2(const at::ArrayRef<bool> &value)
{
    return value.vec();
}

inline aclTensorList *ConvertTypeV2(const std::vector<TensorStructPtr> &at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor *> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertTypeV2(at_tensor_list[i]);
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline std::vector<TensorStructPtr> CopyTypeV2(const at::TensorList &at_tensor_list)
{
    std::vector<TensorStructPtr> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(at_tensor_list[i]);
    }
    return tensor_list;
}

inline std::vector<TensorStructPtr> CopyTypeV2(const TensorListWrapper &tensor_list_wrapper)
{
    std::vector<TensorStructPtr> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = CopyTypeV2(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype});
    }
    return tensor_list;
}

inline aclScalarList *ConvertTypeV2(const std::vector<at::Scalar> &at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar *> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertTypeV2(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline std::vector<at::Scalar> CopyTypeV2(const at::ArrayRef<at::Scalar> &at_scalar_list)
{
    return at_scalar_list.vec();
}

inline TensorStructPtr CopyTypeV2(const c10::optional<at::Tensor> &opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return CopyTypeV2(opt_tensor.value());
    }

    return nullptr;
}

inline aclIntArray *ConvertTypeV2(const c10::optional<std::vector<int64_t>> &opt_array)
{
    if (opt_array.has_value()) {
        return ConvertTypeV2(opt_array.value());
    }

    return nullptr;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::optional<at::IntArrayRef> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalArrayRef<c10::SymInt> &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline c10::optional<std::vector<int64_t>> CopyTypeV2(const c10::OptionalIntArrayRef &opt_array)
{
    if (opt_array.has_value()) {
        return CopyTypeV2(opt_array.value());
    }

    return c10::nullopt;
}

inline aclScalar *ConvertTypeV2(const c10::optional<at::Scalar> &opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertTypeV2(opt_scalar.value());
    }

    return nullptr;
}

inline aclDataType ConvertTypeV2(const at::ScalarType scalarType)
{
    return at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType);
}

inline char* ConvertTypeV2(const std::string &str)
{
    char* string_ptr = const_cast<char *>(str.c_str());
    return string_ptr;
}

inline std::string CopyTypeV2(char* str)
{
    std::string result = str;
    return result;
}

template <typename T> T ConvertTypeV2(T value)
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

template <std::size_t N> void add_param_to_buf(const std::array<bool, N> &value)
{
    MEMCPY_TO_BUF(value.data(), static_cast<int64_t>(value.size() * sizeof(bool)));
}

template <typename T> void add_param_to_buf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void add_param_to_buf(const at::Tensor &);
void add_param_to_buf(const at::Scalar &);
void add_param_to_buf(const at::IntArrayRef &);
void add_param_to_buf(const at::ArrayRef<c10::SymInt> &);
void add_param_to_buf(const at::ArrayRef<bool> &);
void add_param_to_buf(const at::TensorList &);
void add_param_to_buf(const at::ArrayRef<at::Scalar> &);
void add_param_to_buf(const c10::optional<at::Tensor> &);
void add_param_to_buf(const c10::optional<at::IntArrayRef> &);
void add_param_to_buf(const c10::OptionalArrayRef<c10::SymInt> &);
void add_param_to_buf(const c10::OptionalIntArrayRef &);
void add_param_to_buf(const c10::optional<at::Scalar> &);
void add_param_to_buf(const at::ScalarType);
void add_param_to_buf(const string &);
void add_param_to_buf(char *);
void add_param_to_buf(const char *);
void add_param_to_buf(const TensorWrapper &tensor_r);
void add_param_to_buf(const TensorListWrapper &tensor_list_wrapper);
void add_param_to_buf();

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

void add_param_to_buf_v2(TensorStructPtr);
void add_param_to_buf_v2(const at::Scalar &);
void add_param_to_buf_v2(const std::vector<int64_t> &);
void add_param_to_buf_v2(const std::vector<bool> &);
void add_param_to_buf_v2(const std::vector<TensorStructPtr> &);
void add_param_to_buf_v2(const std::vector<at::Scalar> &);
void add_param_to_buf_v2(const c10::optional<std::vector<int64_t>> &);
void add_param_to_buf_v2(const c10::optional<at::Scalar> &);
void add_param_to_buf_v2(const at::ScalarType);
void add_param_to_buf_v2(const string &);
void add_param_to_buf_v2(char *);
void add_param_to_buf_v2(const char *);
void add_param_to_buf_v2();

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
