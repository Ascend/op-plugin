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

#ifndef OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H
#define OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H

#include <unordered_map>
#include <mutex>
#include <memory>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"
#include "OperationCacheCompute.h"


namespace atb {

template <typename ParamType>
class OpParamCache {
public:

    static OpParamCache& getInstance();

    atb::Operation* getOperation(const ParamType& param, const std::string& name);
    atb::Operation* getOperation(uint64_t hashId);
    void saveOperation(uint64_t hashId, atb::Operation* op);

private:
    OpParamCache() = default;

    OpParamCache(const OpParamCache&) = delete;
    OpParamCache& operator=(const OpParamCache&) = delete;

    ~OpParamCache();

    std::unordered_map<uint64_t, atb::Operation*> opMap_;
    mutable std::mutex mutex_;
};


template <typename ParamType>
OpParamCache<ParamType>& OpParamCache<ParamType>::getInstance()
{
    static OpParamCache instance;
    return instance;
}


template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::getOperation(const ParamType& param, const std::string& name)
{
    uint64_t hashValue = computeHash(param);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto opCache = opMap_.find(hashValue);
        if (opCache != opMap_.end()) {
            return opCache->second;
        }
        
        atb::Operation* op = nullptr;
        atb::CreateOperation(param, &op);
        TORCH_CHECK(op != nullptr, name, " CreateOperation failed!");
        opMap_[hashValue] = op;
        return op;
    }
}

template <typename ParamType>
atb::Operation* OpParamCache<ParamType>::getOperation(uint64_t hashId)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto opCache = opMap_.find(hashId);
    if (opCache != opMap_.end()) {
        return opCache->second;
    }

    atb::Operation* op = nullptr;
    return op;
}

template <typename ParamType>
void OpParamCache<ParamType>::saveOperation(uint64_t hashId, atb::Operation* op)
{
    std::lock_guard<std::mutex> lock(mutex_);
    opMap_[hashId] = op;
    return ;
}

template <typename ParamType>
OpParamCache<ParamType>::~OpParamCache()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& opItem: opMap_) {
        DestroyOperation(opItem.second);
    }
}

} // namespace atb

#endif // OPPLUGIN_UTILS_ATB_OPERATION_CREATE_H
