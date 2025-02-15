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

#ifndef OPPLUGIN_UTILS_ATB_PARAM_OPERATION_CACHE_COMPUTE_H
#define OPPLUGIN_UTILS_ATB_PARAM_OPERATION_CACHE_COMPUTE_H

#include <unordered_map>
#include <mutex>
#include <memory>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "op_plugin/third_party/atb/inc/atb_infer.h"


namespace atb {

template <typename T>
struct HashOpParam {
    size_t operator()(const T& param) const;
};


// Each operator implements its own hash function calculation.
// It is possible to hash only the attributes that may change in the parameters of the calculation.
// following example::
//
// `template <>`
// `struct HashOpParam<atb::infer::XXXParam> {   //if XXXParam's transposeA and hasBias need hash`
//     `size_t operator()(const atb::infer::XXXParam& param) const {`
//         `size_t h = std::hash<std::string>{}("transposeA") ^ std::hash<int>{}(param.transposeA);`
//         `h ^= std::hash<std::string>{}("hasBias") ^ std::hash<int>{}(param.hasBias);`
//         `return h;`
//     `}`
// `};`


template <>
struct HashOpParam<atb::infer::LinearParam> {
    size_t operator()(const atb::infer::LinearParam& param) const
    {
        size_t h = 0;
        return h;
    }
};


template <typename T>
size_t computeHash(const T& obj)
{
    return HashOpParam<T>{}(obj);
}

} // namespace atb

#endif // OPPLUGIN_UTILS_ATB_PARAM_OPERATION_CACHE_COMPUTE_H
