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


#ifndef __TORCH_NPU_OP_PLUGIN_UTILS_ASD_SIP_NPU_OP_API__
#define __TORCH_NPU_OP_PLUGIN_UTILS_ASD_SIP_NPU_OP_API__

#include <fstream>
#include <dlfcn.h>
#include <vector>
#include <list>
#include <ATen/Tensor.h>
#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

constexpr int TENSOR_FORMAT_ND = 2;
constexpr int FFT_FAILED = 1;
constexpr size_t MAX_SVECTOR_SIZE = 256;
constexpr size_t DEFAULT_SVECTOR_SIZE = 48;

template <class T, std::size_t MAX_SIZE = DEFAULT_SVECTOR_SIZE> class SVector {
public:
    constexpr SVector() : size_(0)
    {
        static_assert(MAX_SIZE > 0 && MAX_SIZE <= MAX_SVECTOR_SIZE);
        for (std::size_t i = 0; i < MAX_SIZE; ++i) {
            storage_[i] = T{};
        }
    }

private:
    T storage_[MAX_SIZE + 1];
    std::size_t size_{0};
};

typedef struct {
    int dtype;
    int format;
    SVector<int64_t> dims;
    SVector<int64_t> strides;
    int64_t offset;
} MkiTensorDesc;
typedef struct {
    MkiTensorDesc desc;
    void *data = nullptr;
    size_t dataSize = 0;
    void *hostData = nullptr;
} MkiTensor;

enum asdFftType {
    ASCEND_FFT_C2C = 0x10,
    ASCEND_FFT_C2R = 0x11,
    ASCEND_FFT_R2C = 0x12,
    ASCEND_STFT_C2C = 0x20,
    ASCEND_STFT_C2R = 0x21,
    ASCEND_STFT_R2C = 0x22
};

enum asdFftDirection {
    ASCEND_FFT_FORWARD = 0x10,
    ASCEND_FFT_BACKWARD = 0x11,
};

enum asdFft1dDimType {
    ASCEND_FFT_HORIZONTAL = 0x10,
    ASCEND_FFT_VERTICAL = 0x11,
};

typedef MkiTensor (*_asdCreateTensor)(void *data, void *hostData, std::vector<int64_t> dims, int dtype, int format, int size);

typedef void *asdFftHandle;
typedef int (*_asdFftCreate)(asdFftHandle &handle);
typedef int (*_asdFftDestroy)(asdFftHandle handle);
typedef int (*_asdFftSetStream)(asdFftHandle handle, void *stream);
typedef int (*_asdFftSynchronize)(asdFftHandle handle);
typedef int (*_asdFftGetWorkspaceSize)(asdFftHandle handle, size_t &workSize);
typedef int (*_asdFftSetWorkspace)(asdFftHandle handle, void *workspace);
typedef int (*_asdFftMakePlan1D)(asdFftHandle handle, int64_t fftSize, asdFftType fftType, asdFftDirection direction,
                                 int64_t batchSize, asdFft1dDimType dimType);
typedef int (*_asdFftMakePlan2D)(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType,
                                 asdFftDirection direction, int32_t batchSize);

using FftExecApiFunc = int (*)(asdFftHandle handle, const aclTensor *inData, const aclTensor *outData);

#define GET_SIP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetAsdSipApiFuncAddr(#apiName))

inline const char *GetAsdSipApiLibName(void)
{
    return "libasdsip.so";
}

inline void *GetAsdSipApiFuncAddr(const char *apiName)
{
    static auto opApiHandler = dlopen(GetAsdSipApiLibName(), RTLD_LAZY);
    if (opApiHandler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", GetAsdSipApiLibName(), dlerror());
        return nullptr;
    }
    
    auto funcAddr = dlsym(opApiHandler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, GetAsdSipApiLibName(), dlerror());
    }

    return funcAddr;
}

// MKI::TensorDType is same with aclDataType;
inline int ConvertDataType(const at::ScalarType scalarType)
{
    return int(at_npu::native::OpPreparation::convert_to_acl_data_type(scalarType));
}

inline MkiTensor ConvertMkiTensor(const at::Tensor &at_tensor)
{
    if (!at_tensor.defined()) {
        TORCH_CHECK(false, "at_tensor not defined!", OPS_ERROR(ErrCode::PARAM));
    }

    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));

    TORCH_CHECK(at_tensor.is_contiguous(), "Expected tensor is contiguous", OPS_ERROR(ErrCode::PARAM));

    static const auto asdCreateTensor = GET_SIP_API_FUNC(asdCreateTensor);
    if (asdCreateTensor == nullptr) {
        TORCH_CHECK(false, "Invalid asdCreateTensor API!", OPS_ERROR(ErrCode::VALUE));
    }

    int dtype = ConvertDataType(at_tensor.scalar_type());
    auto sip_tensor = asdCreateTensor(const_cast<void *>(at_tensor.storage().data()), nullptr, at_tensor.sizes().vec(),
                                      dtype, TENSOR_FORMAT_ND, at_tensor.storage().nbytes());
    return sip_tensor;
}

inline int asdSipFftCreate(asdFftHandle &handle)
{
    static const auto asdFftCreate = GET_SIP_API_FUNC(asdFftCreate);
    if (asdFftCreate == nullptr) {
        return FFT_FAILED;
    }
    return asdFftCreate(handle);
}

inline int asdSipFftDestroy(asdFftHandle handle)
{
    static const auto asdFftDestroy = GET_SIP_API_FUNC(asdFftDestroy);
    if (asdFftDestroy == nullptr) {
        return FFT_FAILED;
    }
    return asdFftDestroy(handle);
}

inline int asdSipFftSetStream(asdFftHandle handle, void *stream)
{
    static const auto asdFftSetStream = GET_SIP_API_FUNC(asdFftSetStream);
    if (asdFftSetStream == nullptr) {
        return FFT_FAILED;
    }
    return asdFftSetStream(handle, stream);
}

inline int asdSipFftSynchronize(asdFftHandle handle)
{
    static const auto asdFftSynchronize = GET_SIP_API_FUNC(asdFftSynchronize);
    if (asdFftSynchronize == nullptr) {
        return FFT_FAILED;
    }
    return asdFftSynchronize(handle);
}

inline int asdSipFftGetWorkspaceSize(asdFftHandle handle, size_t &workSize)
{
    static const auto asdFftGetWorkspaceSize = GET_SIP_API_FUNC(asdFftGetWorkspaceSize);
    if (asdFftGetWorkspaceSize == nullptr) {
        return FFT_FAILED;
    }
    return asdFftGetWorkspaceSize(handle, workSize);
}

inline int asdSipFftSetWorkspace(asdFftHandle handle, void *workspace)
{
    static const auto asdFftSetWorkspace = GET_SIP_API_FUNC(asdFftSetWorkspace);
    if (asdFftSetWorkspace == nullptr) {
        return FFT_FAILED;
    }
    return asdFftSetWorkspace(handle, workspace);
}

inline int asdSipFftMakePlan1D(asdFftHandle handle, int64_t fftSize, asdFftType fftType, asdFftDirection direction,
                               int64_t batchSize, asdFft1dDimType dimType)
{
    static const auto asdFftMakePlan1D = GET_SIP_API_FUNC(asdFftMakePlan1D);
    if (asdFftMakePlan1D == nullptr) {
        return FFT_FAILED;
    }
    return asdFftMakePlan1D(handle, fftSize, fftType, direction, batchSize, dimType);
}

inline int asdSipFftMakePlan2D(asdFftHandle handle, int64_t fftSizeX, int64_t fftSizeY, asdFftType fftType,
                               asdFftDirection direction, int64_t batchSize)
{
    static const auto asdFftMakePlan2D = GET_SIP_API_FUNC(asdFftMakePlan2D);
    if (asdFftMakePlan2D == nullptr) {
        return FFT_FAILED;
    }
    return asdFftMakePlan2D(handle, fftSizeX, fftSizeY, fftType, direction, batchSize);
}

struct FFTParam {
    int64_t fftXSize = 0;
    int64_t fftYSize = 0;
    asdFftType fftType = asdFftType::ASCEND_FFT_C2C;
    asdFftDirection direction = asdFftDirection::ASCEND_FFT_FORWARD;
    int64_t batchSize = 0;
    asdFft1dDimType dimType = asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
};

inline bool operator==(const FFTParam &one, const FFTParam &other)
{
    return one.fftXSize == other.fftXSize
        && one.fftYSize == other.fftYSize
        && one.fftType == other.fftType
        && one.direction == other.direction
        && one.batchSize == other.batchSize
        && one.dimType == other.dimType;
}

inline asdFftHandle createHandle(const FFTParam &param)
{
    asdFftHandle handle;
    asdSipFftCreate(handle);
    if (param.fftYSize == 0) {
        asdSipFftMakePlan1D(handle, param.fftXSize, param.fftType, param.direction, param.batchSize, param.dimType);
    } else {
        asdSipFftMakePlan2D(handle, param.fftXSize, param.fftYSize, param.fftType, param.direction, param.batchSize);
    }
    return handle;
}

inline void destoryHandle(asdFftHandle handle)
{
    asdSipFftSynchronize(handle);
    asdSipFftDestroy(handle);
}

#endif //  __TORCH_NPU_OP_PLUGIN_UTILS_ASD_SIP_NPU_OP_API__
