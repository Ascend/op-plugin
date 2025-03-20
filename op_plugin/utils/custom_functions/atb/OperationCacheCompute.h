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

constexpr int g_hash_buf_size = 8192;
constexpr int g_hash_buf_max_size = g_hash_buf_size + 1024;
extern thread_local char g_hash_buf[g_hash_buf_size];
extern thread_local int g_hash_offset;

#define MEMCPY_TO_BUF(data_expression, size_expression)                                                                \
    if (g_hash_offset + (size_expression) > g_hash_buf_size) {                                                         \
        g_hash_offset = g_hash_buf_max_size;                                                                           \
        return;                                                                                                        \
    }                                                                                                                  \
    memcpy(g_hash_buf + g_hash_offset, data_expression, size_expression);                                              \
    g_hash_offset += size_expression;

uint64_t calc_hash_id();

template <typename T> void add_param_to_buf(const T &value)
{
    MEMCPY_TO_BUF(&value, sizeof(T));
}

void add_param_to_buf(const string &s);

template <typename T> void add_param_to_buf(const std::string &name, const T &value)
{
    add_param_to_buf(name);
    add_param_to_buf(value);
}

template <typename T>
struct HashOpParam {
    void operator()(const T& param) const {};
};

// Each operator implements its own hash function calculation.
// If the operator parameters do not change, implementation can be omitted.
// It is possible to hash only the attributes that may change in the parameters of the calculation.
// following example::
//
// `template <>`
// `struct HashOpParam<atb::infer::XXXParam> {   //if XXXParam's transposeA and hasBias need hash`
//     `void operator()(const atb::infer::XXXParam& param) const {`
//         `add_param_to_buf("transposeA", param.transposeA);`
//         `add_param_to_buf("hasBias", param.hasBias);`
//     `}`
// `};`

template <>
struct HashOpParam<atb::infer::RmsNormParam> {
    void operator()(const atb::infer::RmsNormParam& param) const
    {
        add_param_to_buf("epsilon", param.normParam.epsilon);
        add_param_to_buf("layerType", param.layerType);
        add_param_to_buf("quantType", param.normParam.quantType);
    }
};

template <>
struct HashOpParam<atb::infer::GroupTopkParam> {
    void operator()(const atb::infer::GroupTopkParam& param) const
    {
        add_param_to_buf("groupNum", param.groupNum);
        add_param_to_buf("k", param.k);
        add_param_to_buf("groupMultiFlag", param.groupMultiFlag);
        add_param_to_buf("n", param.n);
    }
};

template <>
struct HashOpParam<atb::infer::PagedAttentionParam> {
    void operator()(const atb::infer::PagedAttentionParam& param) const
    {
        add_param_to_buf("num_kv_heads", param.kvHeadNum);
        add_param_to_buf("num_heads", param.headNum);
        add_param_to_buf("scale_value", param.qkScale);
        add_param_to_buf("quant_type", param.quantType);
        add_param_to_buf("outdata_type", param.outDataType);
        add_param_to_buf("mla_vheadsize", param.mlaVHeadSize);
        add_param_to_buf("maskType", param.maskType);
        add_param_to_buf("calcType", param.calcType);
    }
};

template <>
struct HashOpParam<atb::infer::SelfAttentionParam> {
    void operator()(const atb::infer::SelfAttentionParam& param) const
    {
        add_param_to_buf("num_kv_heads", param.kvHeadNum);
        add_param_to_buf("num_heads", param.headNum);
        add_param_to_buf("scale_value", param.qkScale);
        add_param_to_buf("calcType", param.calcType);
        add_param_to_buf("kernelType", param.kernelType);
        add_param_to_buf("maskType", param.maskType);
        add_param_to_buf("quantType", param.quantType);
        add_param_to_buf("isTriuMask", param.isTriuMask);
    }
};

template <>
struct HashOpParam<atb::infer::RopeParam> {
    void operator()(const atb::infer::RopeParam& param) const
    {
        add_param_to_buf("rotaryCoeff", param.rotaryCoeff);
    }
};

template <>
struct HashOpParam<atb::infer::ReshapeAndCacheParam> {
    void operator()(const atb::infer::ReshapeAndCacheParam& param) const
    {
        add_param_to_buf("compressType", param.compressType);
        add_param_to_buf("kvCacheCfg", param.kvCacheCfg);
    }
};

template <typename T>
uint64_t computeHash(const T& obj)
{
    g_hash_offset = 0;
    HashOpParam<T>{}(obj);
    return calc_hash_id();
}

} // namespace atb

#endif // OPPLUGIN_UTILS_ATB_PARAM_OPERATION_CACHE_COMPUTE_H
