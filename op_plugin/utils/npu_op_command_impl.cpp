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

#include <tuple>
#include <vector>
#include <memory>
#include <string>
#include "op_plugin/utils/npu_op_command.h"
#include "op_plugin/utils/op_api_common.h"
#include "op_plugin/utils/op_log.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace op_plugin {

namespace {

struct ConvertedParamsHolder {
    std::vector<aclTensor*> acl_tensors;
    std::vector<aclScalar*> acl_scalars;
    std::vector<aclIntArray*> acl_int_arrays;
    std::vector<aclBoolArray*> acl_bool_arrays;
    std::vector<aclTensorList*> acl_tensor_lists;
    std::vector<aclScalarList*> acl_scalar_lists;
    std::vector<TensorStructPtr> tensor_structs;
    std::vector<std::vector<int64_t>> int_vectors;
    std::vector<std::vector<bool>> bool_vectors;
    std::vector<std::vector<TensorStructPtr>> tensor_list_structs;
    std::vector<std::vector<at::Scalar>> scalar_vectors;
    std::vector<c10::optional<std::vector<int64_t>>> opt_int_vectors;
    std::vector<std::string> string_storage;
    
    ~ConvertedParamsHolder()
    {
        for (auto* p : acl_tensors) { Release(p); }
        for (auto* p : acl_scalars) { Release(p); }
        for (auto* p : acl_int_arrays) { Release(p); }
        for (auto* p : acl_bool_arrays) { Release(p); }
        for (auto* p : acl_tensor_lists) { Release(p); }
        for (auto* p : acl_scalar_lists) { Release(p); }
    }
};

std::string ConvertNpuParamToLogInfo(const NpuParam& param, const std::string& param_name)
{
    std::stringstream ss;
    ss << param_name << ": ";
    
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            ss << op_plugin::logging::convert_info(*tensor);
            break;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            ss << op_plugin::logging::convert_info(*scalar);
            break;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            ss << op_plugin::logging::convert_info(*array);
            break;
        }
        case kBool: {
            auto* val = static_cast<const bool*>(param.data);
            ss << op_plugin::logging::convert_info(*val);
            break;
        }
        case kInt64: {
            auto* val = static_cast<const int64_t*>(param.data);
            ss << op_plugin::logging::convert_info(*val);
            break;
        }
        case kDouble: {
            auto* val = static_cast<const double*>(param.data);
            ss << op_plugin::logging::convert_info(*val);
            break;
        }
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            ss << op_plugin::logging::convert_info(*tensor_list);
            break;
        }
        case kScalarList:
        case kScalarArrayRef: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            ss << op_plugin::logging::convert_info(*scalar_list);
            break;
        }
        case kString: {
            auto* str = static_cast<const std::string*>(param.data);
            ss << "string: " << *str << "\n";
            break;
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            ss << op_plugin::logging::convert_info(scalar_type);
            break;
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            ss << op_plugin::logging::convert_info(*opt_tensor);
            break;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            ss << op_plugin::logging::convert_info(*opt_array);
            break;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            ss << op_plugin::logging::convert_info(*opt_scalar);
            break;
        }
        case kBoolArray:
        case kBoolArrayRef: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            ss << op_plugin::logging::convert_info(*array);
            break;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            ss << op_plugin::logging::convert_info(*array);
            break;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            ss << op_plugin::logging::convert_info(*opt_array);
            break;
        }
        case kNullptr:
            ss << "nullptr\n";
            break;
        case kFloat: {
            auto* val = static_cast<const float*>(param.data);
            ss << "float: " << *val << "\n";
            break;
        }
        case kInt32: {
            auto* val = static_cast<const int32_t*>(param.data);
            ss << "int32_t: " << *val << "\n";
            break;
        }
        case kUInt8: {
            auto* val = static_cast<const uint8_t*>(param.data);
            ss << "uint8_t: " << static_cast<int>(*val) << "\n";
            break;
        }
        default:
            ss << "unknown type\n";
            break;
    }
    return ss.str();
}

std::vector<std::string> ParseParamNames(const char* param_names)
{
    std::vector<std::string> result;
    if (param_names == nullptr || param_names[0] == '\0') {
        return result;
    }
    std::string s(param_names);
    size_t pos = 0;
    size_t next;
    
    while ((next = s.find(", ", pos)) != std::string::npos) {
        result.push_back(s.substr(pos, next - pos));
        pos = next + 2;
    }
    if (pos < s.size()) {
        result.push_back(s.substr(pos));
    }
    return result;
}

std::string GenerateNpuParamsLogInfo(const NpuParam* params, int nparams, const char* param_names)
{
    auto names = ParseParamNames(param_names);
    std::string log_info = "\n";
    for (int i = 0; i < nparams; ++i) {
        std::string param_name = (i < static_cast<int>(names.size())) ? names[i] : ("param_" + std::to_string(i));
        log_info += ConvertNpuParamToLogInfo(params[i], param_name);
    }
    return log_info;
}

void AddNpuParamToHashBuf(const NpuParam& param)
{
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            add_param_to_buf(*tensor);
            break;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            add_param_to_buf(*scalar);
            break;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            add_param_to_buf(*array);
            break;
        }
        case kBool: {
            auto* val = static_cast<const bool*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        case kInt64: {
            auto* val = static_cast<const int64_t*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        case kDouble: {
            auto* val = static_cast<const double*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            add_param_to_buf(*tensor_list);
            break;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            add_param_to_buf(*scalar_list);
            break;
        }
        case kString: {
            auto* str = static_cast<const std::string*>(param.data);
            add_param_to_buf(*str);
            break;
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            add_param_to_buf(scalar_type);
            break;
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            add_param_to_buf(*opt_tensor);
            break;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            add_param_to_buf(*opt_array);
            break;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            add_param_to_buf(*opt_scalar);
            break;
        }
        case kBoolArray:
        case kBoolArrayRef: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            add_param_to_buf(*array);
            break;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            add_param_to_buf(*array);
            break;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            add_param_to_buf(*opt_array);
            break;
        }
        case kNullptr:
            add_param_to_buf();
            break;
        case kScalarArrayRef: {
            auto* array = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            add_param_to_buf(*array);
            break;
        }
        case kFloat: {
            auto* val = static_cast<const float*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        case kInt32: {
            auto* val = static_cast<const int32_t*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        case kUInt8: {
            auto* val = static_cast<const uint8_t*>(param.data);
            add_param_to_buf(*val);
            break;
        }
        default:
            break;
    }
}

void AddNpuParamToHashBufV2(const NpuParam& param, ConvertedParamsHolder& holder)
{
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            auto tensor_struct = CopyTypeV2(*tensor);
            holder.tensor_structs.push_back(tensor_struct);
            add_param_to_buf_v2(tensor_struct);
            break;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            add_param_to_buf_v2(*scalar);
            break;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            holder.int_vectors.push_back(array->vec());
            add_param_to_buf_v2(holder.int_vectors.back());
            break;
        }
        case kBool: {
            auto* val = static_cast<const bool*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        case kInt64: {
            auto* val = static_cast<const int64_t*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        case kDouble: {
            auto* val = static_cast<const double*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            std::vector<TensorStructPtr> tensor_struct_list;
            for (const auto& tensor : *tensor_list) {
                auto tensor_struct = CopyTypeV2(tensor);
                holder.tensor_structs.push_back(tensor_struct);
                tensor_struct_list.push_back(tensor_struct);
            }
            holder.tensor_list_structs.push_back(tensor_struct_list);
            add_param_to_buf_v2(tensor_struct_list);
            break;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            holder.scalar_vectors.push_back(scalar_list->vec());
            add_param_to_buf_v2(holder.scalar_vectors.back());
            break;
        }
        case kString: {
            auto* str = static_cast<const std::string*>(param.data);
            add_param_to_buf_v2(*str);
            break;
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            add_param_to_buf_v2(scalar_type);
            break;
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            if (opt_tensor->has_value() && opt_tensor->value().defined()) {
                auto tensor_struct = CopyTypeV2(opt_tensor->value());
                holder.tensor_structs.push_back(tensor_struct);
                add_param_to_buf_v2(tensor_struct);
            } else {
                add_param_to_buf_v2(TensorStructPtr(nullptr));
            }
            break;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(opt_array->value().vec());
                add_param_to_buf_v2(holder.opt_int_vectors.back());
            } else {
                add_param_to_buf_v2(c10::optional<std::vector<int64_t>>());
            }
            break;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            add_param_to_buf_v2(*opt_scalar);
            break;
        }
        case kBoolArray:
        case kBoolArrayRef: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            holder.bool_vectors.push_back(array->vec());
            add_param_to_buf_v2(holder.bool_vectors.back());
            break;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            holder.int_vectors.push_back(CopyTypeV2(*array));
            add_param_to_buf_v2(holder.int_vectors.back());
            break;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(CopyTypeV2(opt_array->value()));
                add_param_to_buf_v2(holder.opt_int_vectors.back());
            } else {
                add_param_to_buf_v2(c10::optional<std::vector<int64_t>>());
            }
            break;
        }
        case kNullptr:
            add_param_to_buf_v2();
            break;
        case kScalarArrayRef: {
            auto* array = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            holder.scalar_vectors.push_back(array->vec());
            add_param_to_buf_v2(holder.scalar_vectors.back());
            break;
        }
        case kFloat: {
            auto* val = static_cast<const float*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        case kInt32: {
            auto* val = static_cast<const int32_t*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        case kUInt8: {
            auto* val = static_cast<const uint8_t*>(param.data);
            add_param_to_buf_v2(*val);
            break;
        }
        default:
            break;
    }
}

bool HitCacheForNpuCmd(aclrtStream acl_stream, const char* api_name, void* opApiFuncAddr,
                       const NpuParam* params, int nparams)
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
    bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(api_name);
    if (!has_func || !can_use) {
        return false;
    }
    uint64_t workspace_size = 0;
    uint64_t* workspace_size_addr = &workspace_size;
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
    add_param_to_buf(std::string(api_name));
    for (int i = 0; i < nparams; ++i) {
        AddNpuParamToHashBuf(params[i]);
    }
    add_param_to_buf(device);
    add_param_to_buf((uintptr_t)acl_stream);
    uint64_t hashId = calc_hash_id();
    setPTAHashKeyFunc(hashId);
    aclOpExecutor* executor = ptaGetExecCacheFunc(hashId, workspace_size_addr);
    if (executor == nullptr) {
        return false;
    }
    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }
    auto acl_call = [workspace_addr, workspace_size, acl_stream, executor, opApiFuncAddr]() -> int {
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        NPU_CHECK_ERROR(api_ret, "call failed");
        return api_ret;
    };
    at_npu::native::OpCommand::RunOpApiV2(api_name, acl_call);
    UnInitCacheThreadLocal();
    return true;
}

bool HitCacheV2ForNpuCmd(aclrtStream acl_stream, const char* api_name, void* opApiFuncAddr,
                         const NpuParam* params, int nparams, int* api_ret,
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
    bool can_use = canUsePTACacheFunc && canUsePTACacheFunc(api_name);
    if (!has_func || !can_use) {
        return false;
    }
    uint64_t workspace_size = 0;
    uint64_t* workspace_size_addr = &workspace_size;
    initPTACacheThreadLocalFunc();
    g_hash_offset = 0;
    add_param_to_buf_v2(deterministic_status);
    if (aic_num != UINT32_MAX && aiv_num != UINT32_MAX) {
        add_param_to_buf_v2(aic_num);
        add_param_to_buf_v2(aiv_num);
    }
    add_param_to_buf_v2(std::string(api_name));
    ConvertedParamsHolder holder;
    for (int i = 0; i < nparams; ++i) {
        AddNpuParamToHashBufV2(params[i], holder);
    }
    add_param_to_buf_v2((uintptr_t)acl_stream);
    if (g_hash_offset == g_hash_buf_max_size) {
        setPTACacheHashKeyFunc(nullptr, 0);
    } else {
        setPTACacheHashKeyFunc(reinterpret_cast<uint8_t*>(g_hash_buf), g_hash_offset);
    }
    aclOpExecutor* executor = ptaFindExecCacheFunc(reinterpret_cast<uint8_t*>(g_hash_buf),
        g_hash_offset, workspace_size_addr);
    if (executor == nullptr) {
        return false;
    }
    void* workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size != 0) {
        workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
    }
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
    *api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    NPU_CHECK_ERROR(*api_ret, "call failed");
    UnInitCacheThreadLocal();
    return true;
}

void* ConvertNpuParam(const NpuParam& param, ConvertedParamsHolder& holder)
{
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            auto* acl_tensor = ConvertType(*tensor);
            holder.acl_tensors.push_back(acl_tensor);
            return acl_tensor;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            auto* acl_scalar = ConvertType(*scalar);
            holder.acl_scalars.push_back(acl_scalar);
            return acl_scalar;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kBool:
            return const_cast<void*>(param.data);
        case kInt64:
            return const_cast<void*>(param.data);
        case kDouble:
            return const_cast<void*>(param.data);
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            auto* acl_tensor_list = ConvertType(*tensor_list);
            holder.acl_tensor_lists.push_back(acl_tensor_list);
            return acl_tensor_list;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            auto* acl_scalar_list = ConvertType(*scalar_list);
            holder.acl_scalar_lists.push_back(acl_scalar_list);
            return acl_scalar_list;
        }
        case kString: {
            const std::string* str = static_cast<const std::string*>(param.data);
            holder.string_storage.push_back(*str);
            return holder.string_storage.back().data();
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            auto acl_dtype = ConvertType(scalar_type);
            return reinterpret_cast<void*>(static_cast<intptr_t>(acl_dtype));
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            if (opt_tensor->has_value() && opt_tensor->value().defined()) {
                auto* acl_tensor = ConvertType(opt_tensor->value());
                holder.acl_tensors.push_back(acl_tensor);
                return acl_tensor;
            }
            return nullptr;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            if (opt_array->has_value()) {
                auto* acl_array = ConvertType(opt_array->value());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            if (opt_scalar->has_value()) {
                auto* acl_scalar = ConvertType(opt_scalar->value());
                holder.acl_scalars.push_back(acl_scalar);
                return acl_scalar;
            }
            return nullptr;
        }
        case kBoolArray: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            if (opt_array->has_value()) {
                auto* acl_array = ConvertType(opt_array->value());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kNullptr:
            return nullptr;
        case kScalarArrayRef: {
            auto* array = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_scalar_lists.push_back(acl_array);
            return acl_array;
        }
        case kBoolArrayRef: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            auto* acl_array = ConvertType(*array);
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kFloat:
            return const_cast<void*>(param.data);
        case kInt32:
            return const_cast<void*>(param.data);
        case kUInt8:
            return const_cast<void*>(param.data);
        default:
            return nullptr;
    }
}

void* CopyNpuParamV2(const NpuParam& param, ConvertedParamsHolder& holder)
{
    switch (param.type) {
        case kTensor: {
            auto* tensor = static_cast<const at::Tensor*>(param.data);
            auto tensor_struct = CopyTypeV2(*tensor);
            holder.tensor_structs.push_back(tensor_struct);
            auto* acl_tensor = ConvertTypeV2(tensor_struct);
            holder.acl_tensors.push_back(acl_tensor);
            return acl_tensor;
        }
        case kScalar: {
            auto* scalar = static_cast<const at::Scalar*>(param.data);
            auto* acl_scalar = ConvertTypeV2(*scalar);
            holder.acl_scalars.push_back(acl_scalar);
            return acl_scalar;
        }
        case kIntArrayRef: {
            auto* array = static_cast<const at::IntArrayRef*>(param.data);
            holder.int_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.int_vectors.back());
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kBool:
            return const_cast<void*>(param.data);
        case kInt64:
            return const_cast<void*>(param.data);
        case kDouble:
            return const_cast<void*>(param.data);
        case kTensorList: {
            auto* tensor_list = static_cast<const at::TensorList*>(param.data);
            std::vector<TensorStructPtr> tensor_struct_list;
            for (const auto& tensor : *tensor_list) {
                auto tensor_struct = CopyTypeV2(tensor);
                holder.tensor_structs.push_back(tensor_struct);
                tensor_struct_list.push_back(tensor_struct);
            }
            holder.tensor_list_structs.push_back(tensor_struct_list);
            auto* acl_tensor_list = ConvertTypeV2(tensor_struct_list);
            holder.acl_tensor_lists.push_back(acl_tensor_list);
            return acl_tensor_list;
        }
        case kScalarList: {
            auto* scalar_list = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            holder.scalar_vectors.push_back(scalar_list->vec());
            auto* acl_scalar_list = ConvertTypeV2(holder.scalar_vectors.back());
            holder.acl_scalar_lists.push_back(acl_scalar_list);
            return acl_scalar_list;
        }
        case kString: {
            const std::string* str = static_cast<const std::string*>(param.data);
            holder.string_storage.push_back(*str);
            return holder.string_storage.back().data();
        }
        case kScalarType: {
            auto scalar_type = *static_cast<const at::ScalarType*>(param.data);
            auto acl_dtype = ConvertTypeV2(scalar_type);
            return reinterpret_cast<void*>(static_cast<intptr_t>(acl_dtype));
        }
        case kOptionalTensor: {
            auto* opt_tensor = static_cast<const c10::optional<at::Tensor>*>(param.data);
            if (opt_tensor->has_value() && opt_tensor->value().defined()) {
                auto tensor_struct = CopyTypeV2(opt_tensor->value());
                holder.tensor_structs.push_back(tensor_struct);
                auto* acl_tensor = ConvertTypeV2(tensor_struct);
                holder.acl_tensors.push_back(acl_tensor);
                return acl_tensor;
            }
            return nullptr;
        }
        case kOptionalIntArrayRef: {
            auto* opt_array = static_cast<const c10::optional<at::IntArrayRef>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(opt_array->value().vec());
                auto* acl_array = ConvertTypeV2(holder.opt_int_vectors.back());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kOptionalScalar: {
            auto* opt_scalar = static_cast<const c10::optional<at::Scalar>*>(param.data);
            if (opt_scalar->has_value()) {
                auto* acl_scalar = ConvertTypeV2(opt_scalar->value());
                holder.acl_scalars.push_back(acl_scalar);
                return acl_scalar;
            }
            return nullptr;
        }
        case kBoolArray: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            holder.bool_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.bool_vectors.back());
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kSymIntArrayRef: {
            auto* array = static_cast<const at::ArrayRef<c10::SymInt>*>(param.data);
            holder.int_vectors.push_back(CopyTypeV2(*array));
            auto* acl_array = ConvertTypeV2(holder.int_vectors.back());
            holder.acl_int_arrays.push_back(acl_array);
            return acl_array;
        }
        case kOptionalSymIntArrayRef: {
            auto* opt_array = static_cast<const c10::OptionalArrayRef<c10::SymInt>*>(param.data);
            if (opt_array->has_value()) {
                holder.opt_int_vectors.push_back(CopyTypeV2(opt_array->value()));
                auto* acl_array = ConvertTypeV2(holder.opt_int_vectors.back());
                holder.acl_int_arrays.push_back(acl_array);
                return acl_array;
            }
            return nullptr;
        }
        case kNullptr:
            return nullptr;
        case kScalarArrayRef: {
            auto* array = static_cast<const at::ArrayRef<at::Scalar>*>(param.data);
            holder.scalar_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.scalar_vectors.back());
            holder.acl_scalar_lists.push_back(acl_array);
            return acl_array;
        }
        case kBoolArrayRef: {
            auto* array = static_cast<const at::ArrayRef<bool>*>(param.data);
            holder.bool_vectors.push_back(array->vec());
            auto* acl_array = ConvertTypeV2(holder.bool_vectors.back());
            holder.acl_bool_arrays.push_back(acl_array);
            return acl_array;
        }
        case kFloat:
            return const_cast<void*>(param.data);
        case kInt32:
            return const_cast<void*>(param.data);
        case kUInt8:
            return const_cast<void*>(param.data);
        default:
            return nullptr;
    }
}

int CallGetWorkspaceSize(void* func_addr, const std::vector<void*>& params)
{
    using GetWorkspaceSizeFunc = int (*)(...);
    auto func = reinterpret_cast<GetWorkspaceSizeFunc>(func_addr);
    switch (params.size()) {
        case 1: return ((int (*)(void*))func)(params[0]);
        case 2: return ((int (*)(void*, void*))func)(params[0], params[1]);
        case 3: return ((int (*)(void*, void*, void*))func)(params[0], params[1], params[2]);
        case 4: return ((int (*)(void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3]);
        case 5: return ((int (*)(void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4]);
        case 6: return ((int (*)(void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5]);
        case 7: return ((int (*)(void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6]);
        case 8: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
        case 9: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]);
        case 10: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9]);
        case 11: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10]);
        case 12: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11]);
        case 13: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12]);
        case 14: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13]);
        case 15: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14]);
        case 16: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15]);
        case 17: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16]);
        case 18: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17]);
        case 19: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18]);
        case 20: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19]);
        case 21: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20]);
        case 22: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21]);
        case 23: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22]);
        case 24: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23]);
        case 25: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24]);
        case 26: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25]);
        case 27: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26]);
        case 28: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27]);
        case 29: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28]);
        case 30: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29]);
        case 31: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30]);
        case 32: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31]);
        case 33: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32]);
        case 34: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33]);
        case 35: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34]);
        case 36: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35]);
        case 37: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36]);
        case 38: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37]);
        case 39: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38]);
        case 40: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39]);
        case 41: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40]);
        case 42: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41]);
        case 43: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42]);
        case 44: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43]);
        case 45: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44]);
        case 46: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45]);
        case 47: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46]);
        case 48: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47]);
        case 49: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47], params[48]);
        case 50: return ((int (*)(void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))func)(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14], params[15], params[16], params[17], params[18], params[19], params[20], params[21], params[22], params[23], params[24], params[25], params[26], params[27], params[28], params[29], params[30], params[31], params[32], params[33], params[34], params[35], params[36], params[37], params[38], params[39], params[40], params[41], params[42], params[43], params[44], params[45], params[46], params[47], params[48], params[49]);
        default:
            ASCEND_LOGE("Too many parameters: %zu", params.size());
            return -1;
    }
}
}

extern "C" {
TORCH_NPU_API void ExecNpuCmdExtImpl(const char* api_name, const NpuParam* params, int nparams, const char* param_names)
{
    static const auto task_queue_enable = c10_npu::option::OptionsManager::GetTaskQueueEnable();
    
    std::string workspace_api_name = std::string(api_name) + "GetWorkspaceSize";
    auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(workspace_api_name.c_str());
    auto opApiFuncAddr = GetOpApiFuncAddr(api_name);
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        api_name, " or ", workspace_api_name, " not in ", GetOpApiLibName(),
        ", or ", GetOpApiLibName(), " not found.",
        OPS_ERROR(ErrCode::PTR));
    
    if (task_queue_enable == 2) {
        auto params_log_info = GenerateNpuParamsLogInfo(params, nparams, param_names);
        OP_EXEC_LOG_EXT_WITH_TASK_QUEUE(api_name, "EXEC_NPU_CMD_EXT", "2", params_log_info.c_str());
        
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        if (c10_npu::check_enqueue_need_use(acl_stream)) {
            c10_npu::UseStreamResInCurrentThread(acl_stream);
        }
        
        auto deterministic_status = at::globalContext().deterministicAlgorithms();
        uint32_t aic_num = UINT32_MAX;
        uint32_t aiv_num = UINT32_MAX;
        if (c10_npu::is_core_control_enabled()) {
            aic_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
            aiv_num = c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
        }
        
        auto acl_call = [api_name, params, nparams, acl_stream, deterministic_status, aic_num, aiv_num,
                         getWorkspaceSizeFuncAddr, opApiFuncAddr, initMemAddr, unInitMemAddr, releaseMemAddr]() mutable -> int {
            if (c10_npu::check_dequeue_need_use(acl_stream)) {
                c10_npu::UseStreamResInCurrentThread(acl_stream);
            }
            
            int api_ret = 0;
            if (HitCacheV2ForNpuCmd(acl_stream, api_name, opApiFuncAddr, params, nparams,
                                    &api_ret, deterministic_status, aic_num, aiv_num)) {
                return api_ret;
            }
            
            uint64_t workspace_size = 0;
            uint64_t* workspace_size_addr = &workspace_size;
            aclOpExecutor* executor = nullptr;
            aclOpExecutor** executor_addr = &executor;
            InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
            UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
            
            at_npu::native::SetDeterministicOps(deterministic_status);
            if (initMemFunc) {
                initMemFunc(nullptr, false);
            }
            
            ConvertedParamsHolder holder;
            std::vector<void*> copied_params;
            copied_params.reserve(nparams);
            for (int i = 0; i < nparams; ++i) {
                copied_params.push_back(CopyNpuParamV2(params[i], holder));
            }
            
            std::vector<void*> converted_params = copied_params;
            converted_params.push_back(workspace_size_addr);
            converted_params.push_back(executor_addr);
            
            auto workspace_status = CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted_params);
            NPU_CHECK_ERROR(workspace_status, (std::string("call ") + api_name + " failed").c_str());
            
            void* workspace_addr = nullptr;
            at::Tensor workspace_tensor;
            if (workspace_size != 0) {
                workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size, acl_stream);
                workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
            }
            
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
            api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
            NPU_CHECK_ERROR(api_ret, (std::string("call ") + api_name + " failed").c_str());
            
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
            if (releaseMemFunc) {
                releaseMemFunc(nullptr, false);
            }
            if (unInitMemFunc) {
                unInitMemFunc(nullptr, false);
            }
            UnInitCacheThreadLocal();
            
            return api_ret;
        };
        
        at_npu::native::OpCommand::RunOpApiV2(api_name, acl_call);
    } else {
        auto params_log_info = GenerateNpuParamsLogInfo(params, nparams, param_names);
        OP_EXEC_LOG_EXT_WITH_TASK_QUEUE(api_name, "EXEC_NPU_CMD_EXT", "1", params_log_info.c_str());
        
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
        if (c10_npu::check_enqueue_need_use(acl_stream)) {
            c10_npu::UseStreamResInCurrentThread(acl_stream);
        }
        
        if (HitCacheForNpuCmd(acl_stream, api_name, opApiFuncAddr, params, nparams)) {
            return;
        }
        
        uint64_t workspace_size = 0;
        uint64_t* workspace_size_addr = &workspace_size;
        aclOpExecutor* executor = nullptr;
        aclOpExecutor** executor_addr = &executor;
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
        
        at_npu::native::SetDeterministic();
        if (initMemFunc) {
            initMemFunc(nullptr, false);
        }
        
        ConvertedParamsHolder holder;
        std::vector<void*> converted_params;
        converted_params.reserve(nparams + 2);
        for (int i = 0; i < nparams; ++i) {
            converted_params.push_back(ConvertNpuParam(params[i], holder));
        }
        converted_params.push_back(workspace_size_addr);
        converted_params.push_back(executor_addr);
        
        auto workspace_status = CallGetWorkspaceSize(getWorkspaceSizeFuncAddr, converted_params);
        NPU_CHECK_ERROR(workspace_status, (std::string("call ") + api_name + " failed").c_str());
        
        void* workspace_addr = nullptr;
        at::Tensor workspace_tensor;
        if (workspace_size != 0) {
            workspace_tensor = at_npu::native::OpPreparation::unsafe_empty_workspace(workspace_size);
            workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
        }
        
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor,
                         opApiFuncAddr, releaseMemAddr, api_name, holder = std::move(holder)]() mutable -> int {
            if (c10_npu::check_dequeue_need_use(acl_stream)) {
                c10_npu::UseStreamResInCurrentThread(acl_stream);
            }
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
            NPU_CHECK_ERROR(api_ret, (std::string("call ") + api_name + " failed").c_str());
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
            if (releaseMemFunc) {
                releaseMemFunc(nullptr, false);
            }
            return api_ret;
        };
        
        at_npu::native::OpCommand::RunOpApiV2(api_name, acl_call);
        
        if (unInitMemFunc) {
            unInitMemFunc(nullptr, false);
        }
        UnInitCacheThreadLocal();
    }
}
}
}
