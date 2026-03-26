// Copyright (c) 2026 Huawei Technologies Co., Ltd
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

#ifndef OP_PLUGIN_UTILS_NPU_OP_COMMAND_H_
#define OP_PLUGIN_UTILS_NPU_OP_COMMAND_H_

#include <cstdint>
#include <cstddef>
#include <ATen/ATen.h>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace op_plugin {

enum NpuParamType : int32_t {
    kTensor = 0,
    kScalar = 1,
    kIntArrayRef = 2,
    kBool = 3,
    kInt64 = 4,
    kDouble = 5,
    kTensorList = 6,
    kScalarList = 7,
    kString = 8,
    kScalarType = 9,
    kOptionalTensor = 10,
    kOptionalIntArrayRef = 11,
    kOptionalScalar = 12,
    kBoolArray = 13,
    kSymIntArrayRef = 14,
    kOptionalSymIntArrayRef = 15,
    kNullptr = 16,
    kScalarArrayRef = 17,
    kBoolArrayRef = 18,
    kFloat = 19,
    kInt32 = 20,
    kUInt8 = 21,
};

struct NpuParam {
    const void* data;
    NpuParamType type;
};

inline NpuParam MakeNpuParam(const at::Tensor& v)                           { return {&v, kTensor}; }
inline NpuParam MakeNpuParam(const at::Scalar& v)                           { return {&v, kScalar}; }
inline NpuParam MakeNpuParam(const at::IntArrayRef& v)                      { return {&v, kIntArrayRef}; }
inline NpuParam MakeNpuParam(const bool& v)                                 { return {&v, kBool}; }
inline NpuParam MakeNpuParam(const int64_t& v)                              { return {&v, kInt64}; }
inline NpuParam MakeNpuParam(const double& v)                               { return {&v, kDouble}; }
inline NpuParam MakeNpuParam(const std::string& v)                          { return {&v, kString}; }
inline NpuParam MakeNpuParam(const at::ScalarType& v)                       { return {&v, kScalarType}; }
inline NpuParam MakeNpuParam(const c10::optional<at::Tensor>& v)            { return {&v, kOptionalTensor}; }
inline NpuParam MakeNpuParam(const c10::optional<at::IntArrayRef>& v)       { return {&v, kOptionalIntArrayRef}; }
inline NpuParam MakeNpuParam(const c10::optional<at::Scalar>& v)            { return {&v, kOptionalScalar}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<c10::SymInt>& v)            { return {&v, kSymIntArrayRef}; }
inline NpuParam MakeNpuParam(const c10::OptionalArrayRef<c10::SymInt>& v)   { return {&v, kOptionalSymIntArrayRef}; }
inline NpuParam MakeNpuParam(std::nullptr_t)                                { return {nullptr, kNullptr}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<at::Scalar>& v)             { return {&v, kScalarArrayRef}; }
inline NpuParam MakeNpuParam(const at::ArrayRef<bool>& v)                   { return {&v, kBoolArrayRef}; }
inline NpuParam MakeNpuParam(const float& v)                                { return {&v, kFloat}; }
inline NpuParam MakeNpuParam(const int32_t& v)                              { return {&v, kInt32}; }
inline NpuParam MakeNpuParam(const uint8_t& v)                              { return {&v, kUInt8}; }


extern "C" {
TORCH_NPU_API void ExecNpuCmdExtImpl(const char* api_name, const NpuParam* params, int nparams, const char* param_names);
}

template<typename... Args>
inline void ExecNpuCmd(const char* api_name, const char* param_names, const Args&... args)
{
    NpuParam params[] = {MakeNpuParam(args)...};
    ExecNpuCmdExtImpl(api_name, params, sizeof...(Args), param_names);
}

#define EXEC_NPU_CMD_EXT(aclnn_api, ...)                                                                              \
    do {                                                                                                              \
        ::op_plugin::ExecNpuCmd(#aclnn_api, #__VA_ARGS__, __VA_ARGS__);                                               \
    } while (false)

}

#endif
