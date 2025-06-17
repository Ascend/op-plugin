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

#ifndef INC_EXTERNAL_ATB_INFEROPPARAM_H
#define INC_EXTERNAL_ATB_INFEROPPARAM_H
#include <cstdint>
#include <string>
#include <limits>
#include <hccl/hccl_types.h>
#include <acl/acl.h>
#include "./svector.h"

//!
//! \file infer_op_params.h
//!
//! \brief 定义加速库所有推理算子参数
//!

//!
//! \namespace atb
//!
//! \brief 加速库命名空间.
//!
namespace atb {

namespace infer {

//!
//! \enum InputLayout
//!
//! \brief 数据排布类型
//!
enum InputLayout : int {
    TYPE_BSND = 0, //!< 默认值，表示数据排布为BSND
    TYPE_BNSD      //!< 表示数据排布为BNSD
};

//!
//! \enum QuantType
//!
//! \brief 量化支持的类型
//!
enum QuantType : int {
    QUANT_UNDEFINED = 0, //!< 不量化
    QUANT_INT4,          //!< 当前不支持
    QUANT_INT8,          //!< int8量化
    QUANT_INT16,         //!< 当前不支持
    QUANT_FLOAT8,        //!< 当前不支持
    QUANT_FLOAT16,       //!< 当前不支持
};

//!
//! \enum DynamicQuantType
//!
//! \brief 动态量化支持的类型
//!
enum DynamicQuantType : int {
    DYNAMIC_QUANT_UNDEFINED = 0, //!< 非动态量化
    DYNAMIC_QUANT_SYMMETRIC,     //!< 对称动态量化
    DYNAMIC_QUANT_ASYMMETRIC,    //!< 非对称动态量化，暂不支持
};

//!
//! \enum ActivationType
//!
//! \brief 激活支持的类型
//! ACTIVATION_FAST_GELU：快速运算的Gelu激活函数，对Tensor内每个element做Gelu激活函数近似计算，计算速度更快，同时保持较高的准确性。
//! ACTIVATION_SWIGLU_FORWARD: Swiglu正向激活函数。Atlas 推理系列产品中只支持32位对齐的数据。
//! ACTIVATION_FASTER_GELU_FORWARD: 简化后的FastGelu激活函数，计算速度更快。
//! ACTIVATION_SWIGLU_BACKWARD: Swiglu正向激活函数的反向，求梯度时使用。只支持Atlas 800I A2推理产品。
//!
enum ActivationType : int {
    ACTIVATION_UNDEFINED = 0,       //!< 未定义
    ACTIVATION_RELU,                //!< RELU激活类型
    ACTIVATION_GELU,                //!< GELU激活类型
    ACTIVATION_FAST_GELU,           //!< FAST_GELU激活类型
    ACTIVATION_SWISH,               //!< SWISH激活类型
    ACTIVATION_LOG,                 //!< LOG激活类型
    ACTIVATION_SWIGLU_FORWARD,      //!< SWIGLU_FORWARD激活类型
    ACTIVATION_SWIGLU_BACKWARD,     //!< SWIGLU_BACKWARD激活类型
    ACTIVATION_SIGMOID,             //!< SIGMOID激活类型
    ACTIVATION_FASTER_GELU_FORWARD, //!< FASTER_GELU_FORWARD激活类型
    ACTIVATION_MAX,                 //!< 枚举最大值, 非激活类型
};

//!
//! \enum CommMode
//!
//! \brief 通信算子支持的通信模式.
//!
enum CommMode : int {
    COMM_UNDEFINED = -1, //!< 未定义
    COMM_MULTI_PROCESS,  //!< 指定多进程通信
    COMM_MULTI_THREAD,   //!< 指定多线程通信
};

//!
//! \struct RmsNormParam
//!
//! \brief RMS归一化处理。
//!
//! \warning 所有输入输出Tensor的最后一维大小相等。
//! Atlas 推理系列产品中不支持bf16类型数据。
//!
struct RmsNormParam {
    //!
    //! \brief RmsNormType
    //!
    enum RmsNormType : int {
        RMS_NORM_UNDEFINED = 0, //!< 默认值，未定义
        RMS_NORM_NORM,          //!< NORM参数。
        RMS_NORM_PRENORM,       //!< PRENORM参数。
        RMS_NORM_POSTNORM,      //!< POSTNORM参数
    };
    //!
    //! \brief PrecisionMode
    //!
    enum PrecisionMode : int {
        HIGH_PRECISION_MODE = 0, //!< 中间计算使用float类型
        HIGH_PERFORMANCE_MODE,   //!< 中间计算使用float16类型
    };
    //!
    //! \brief ModelType
    //!
    enum ModelType : int {
        LLAMA_MODEL = 0, //!< 默认值，使用Llama rmsnorm的公式
        GEMMA_MODEL,     //!< 使用Gemma rmsnorm的公式
    };
    //!
    //! \brief NormParam
    //!
    struct NormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED, QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief Epsilon，默认为1e-5，暂时不使用。
        double layerNormEps = 1e-5;
        //! \brief 默认为False，设置为true时会使用训练的rmsnormforward算子。仅在Atlas 800I A2推理产品上支持该设置。
        //!  不支持和“precisionMode”，“modelType”同时设置。量化场景下不支持使用“rstd”。
        bool rstd = false;
        //! \brief 默认为HIGH_PRECISION_MODE。
        //! 支持参数如下：
        //! HIGH_PRECISION_MODE：默认值，中间计算使用float类型
        //! HIGH_PERFORMANCE_MODE： 中间计算使用float16类型
        //! 不支持和“rstd”，“modelType”同时设置。输入类型只支持float16。
        //! 量化场景下不支持使用“precisionMode”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        PrecisionMode precisionMode = HIGH_PRECISION_MODE;
        //! \brief 默认为LLAMA_MODEL，设置为GEMMA_MODEL时使用gemma模型的rmsnorm计算公式。
        //! 支持参数如下：
        //! LLAMA_MODEL：默认值， Llama的rms norm计算公式。
        //! GEMMA_MODEL：Gemma的rms norm计算公式。
        //! 不支持和“rstd”，“precisionMode”同时启用。
        //! 量化场景下不支持使用“modelType”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        ModelType modelType = LLAMA_MODEL;
        //! \brief 动态量化类型。默认为DYNAMIC_QUANT_UNDEFINED非动态量化。当前版本暂不支持非对称动态量化。
        DynamicQuantType dynamicQuantType = DYNAMIC_QUANT_UNDEFINED;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[32] = {0};
    };
    //!
    //! \brief PreNormParam
    //!
    struct PreNormParam {
        //! \brief 量化类型。
        //! 当前支持以下类型。
        //! QUANT_UNDEINFED
        //! QUANT_INT8
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。量化场景下不支持使用“hasBias”，该场景下配置该参数将返回报错ERROR_INVALID_PARAM。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //!
    //! \brief PostNormParam
    //!
    struct PostNormParam {
        //! \brief 量化类型。
        //! 当前仅支持QUANT_UNDEINFED。
        QuantType quantType = QUANT_UNDEFINED;
        //! \brief Epsilon，归一化时加在分母上防止除零。
        float epsilon = 1e-5;
        //! \brief 是否叠加偏置。默认为False，当需要输入beta时设置为True。
        bool hasBias = false;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[23] = {0};
    };
    //! \brief 归一化类型，参数如下：
    //! RMS_NORM_UNDEFINED：默认值，未定义。
    //! RMS_NORM_NORM：NORM参数。
    //! RMS_NORM_PRENORM：PRENORM参数。
    //! RMS_NORM_POSTNORM：POSTNORM参数。
    RmsNormType layerType = RMS_NORM_UNDEFINED;
    //! \brief NORM参数。
    NormParam normParam;
    //! \brief PRENORM参数。
    PreNormParam preNormParam;
    //! \brief POSTNORM参数。
    PostNormParam postNormParam;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \struct LinearParam
//!
//! \brief 将A、B两个矩阵进行矩阵乘运算，同时可以选择对矩阵乘的运算结果进行叠加偏置、InplaceAdd融合或反量化操作。
//!
//! \note 算子本质上是接收x和weight两个输入tensor作为A矩阵和B矩阵进行矩阵乘运算，可通过参数transposeA与transposeB控制做矩
//! 阵乘前是否需要对A矩阵和B矩阵进行行列转置，根据参数转置后的A矩阵和B矩阵需满足矩阵乘维度关系。例如，当transposeA为false，
//! transposeB为true时，x和weight的shape可以分别为[m, k]和[n, k]。
//!
//! \note 该算子支持浮点和量化场景，当参数outDataType值为ACL_DT_UNDEFINED时为浮点场景，否则为量化场景。
//!
struct LinearParam {
    //!
    //! \brief 是否转置A矩阵。
    //!
    //! \note 默认值为false，不转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为false。
    //!
    bool transposeA = false;
    //!
    //! \brief 是否转置B矩阵。
    //!
    //! \note 默认值为true，转置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    bool transposeB = true;
    //!
    //! \brief 是否叠加偏置。
    //!
    //! \note 默认值为true，叠加偏置。
    //!
    //! \warning 在量化场景下，非Atlas 800I A2推理产品仅支持配置为true。
    //!
    //! \warning enAccum为true时，仅支持配置为false。
    //!
    bool hasBias = true;
    //!
    //! \brief 输出数据类型。
    //!
    //! \note 默认值为ACL_DT_UNDEFINED。
    //!
    //! \warning 浮点场景下：支持配置为ACL_DT_UNDEFINED。
    //!
    //! \warning 量化场景下：Atlas 800I A2推理产品支持配置为ACL_FLOAT16/ACL_BF16，否则，仅支持配置为ACL_FLOAT16。
    //!
    aclDataType outDataType = ACL_DT_UNDEFINED;
    //!
    //! \brief 是否使能累加。
    //!
    //! \note 默认值为false，不使能累加。
    //!
    //! \warning 仅在Atlas 800I A2推理产品支持配置为true。
    //!
    //! \warning hasBias为true时，仅支持配置为false。
    //!
    //! \warning 量化场景下，仅支持配置为false。
    //!
    bool enAccum = false;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[23] = {0};
};

struct GroupTopkParam {
    //!
    //! \brief 每个token分组数量。注：“专家总数”为inTensor0Desc.shape.dims[1]的值。
    //!
    //! \note 必传，默认值为1，取值范围为[1, 专家总数]。
    //!
    //! \warning groupNum需要保证可以被inTensor0Desc.shape.dims[1]整除。
    //!
    int32_t groupNum = 1;
    //!
    //! \brief 选择top K组数量。
    //!
    //! \note 必传，默认值为0，取值范围为[1, groupNum]。
    //!
    //! \warning
    //!
    int32_t k = 0;
    //!
    //! \enum GroupMultiFlag
    //!
    //! \brief 指定GroupTopk每组中取值计算的方式。
    //!
    //! \warning
    //!
    enum GroupMultiFlag : uint16_t {
        UNDEFINED = 0, //!< 默认方式，每组内取最大值。
        SUM_MULTI_MAX  //!< 每组内取n个最大值求和，需要设置参数n
    };
    //!
    //! \brief 指定GroupTopk每组中取值计算的方式。
    //!
    //! \note 默认值为UNDEFINED。
    //!
    //! \warning 取值为SUM_MULTI_MAX时需要传入参数n。
    //!
    GroupMultiFlag groupMultiFlag = UNDEFINED;
    //!
    //! \brief 每组内取值的个数。
    //!
    //! \note 默认值为1，取值范围为[1,expert_num/groupNum]。
    //!
    //! \warning 只有当groupMultiFlag为SUM_MULTI_MAX时有效
    //!
    uint16_t n = 1;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[12] = {0};
};

//!
//! \brief PagedAttention.
//!
//! 一个Q有多个token，一个token对应多个KV的token，以token0为例，block_table代表其对应的KV的block_id，-1代表截止，
//! 所以第二行和第四行为其目标block，context_lens则表示KV有多少个token，则代表仅有block_id为(3,4,5,9,10)是需要与Q进行计算的。
//!
struct PagedAttentionParam {
    //! query 头大小
    int32_t headNum = 0;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1.0;
    //! kv头数量
    int32_t kvHeadNum = 0;
    //!
    //! \enum MaskType
    //!
    //! \brief The type values of MaskType.
    //!
    enum MaskType : int {
        UNDEFINED = 0,   //!< 默认值，全0的mask
        MASK_TYPE_NORM,  //!< 倒三角mask
        MASK_TYPE_ALIBI, //!< alibi mask
        MASK_TYPE_SPEC   //!< 并行解码mask
    };
    //! mask类型
    MaskType maskType = UNDEFINED;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //!
    //! \enum QuantType
    //!
    //! \brief quant类型
    //!
    enum QuantType : int {
        TYPE_QUANT_UNDEFINED = 0, //!< 默认值，不与量化融合，此
        TYPE_DEQUANT_FUSION,      //!< 与反量化融合, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_OFFLINE,   //!< 离线INT8量化, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_ONLINE     //!< 在线INT8量化, 只支持Atlas 800I A2推理产品
    };
    //!
    //! 量化类型：
    //! 为TYPE_QUANT_UNDEFINED时q，keyCache，valueCache为bf16/float16。
    //! 为TYPE_DEQUANT_FUSION时q为bf16/float16，keyCache，valueCache为int8。
    //! 为TYPE_QUANT_QKV_OFFLINE或TYPE_QUANT_QKV_ONLINE时q，keyCache，valueCache为int8。
    //! keyCache,valueCache的headsize等长，范围为（0, 256]，且block_size * head_size ≤ 128 * 128。
    //! outdatatype需要配置，只能是ACL_FLOAT16或ACL_BF16。inputLayout只支持TYPE_BSND。
    QuantType quantType = TYPE_QUANT_UNDEFINED;

    //! output数据类型（格式为aclDataType）
    aclDataType outDataType = ACL_DT_UNDEFINED;

    //! 开启量化功能后是否使用offset
    bool hasQuantOffset = false;
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD, //!< 压缩key_cache, value_cache的kvHead维度, 只支持Atlas 800I A2推理产品。
        COMPRESS_TYPE_KVHEAD_ROPE, //!< rope场景压缩key_cache, value_cache的kvHead维度, 只支持Atlas 800I A2推理产品。
        COMPRESS_TYPE_MAX //!< 压缩类型边界值，仅用于判断是否出界，所有情况不能取该值。
    };
    //!
    //! 压缩方式
    //! 为COMPRESS_TYPE_KVHEAD时，不支持quanttype为2和3。
    //! 为COMPRESS_TYPE_KVHEAD_ROPE时, maskType需传0。不支持quanttype为2和3。
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //!
    //! \enum CalcType
    //!
    //! \brief The type values of CalcType.
    //!
    enum CalcType : int {
        CALC_TYPE_UNDEFINED = 0, //!< 默认值，不开启并行解码
        CALC_TYPE_SPEC           //!< 并行解码功能，此时只支持quantType = 0
    };
    //! 计算类型
    CalcType calcType = CALC_TYPE_UNDEFINED;

    //!
    //! \enum ScaleType
    //!
    //! \brief The type values of ScaleType.
    //!
    enum ScaleType : int {
        SCALE_TYPE_TOR = 0, //!< 默认值，不开启LogN缩放
        SCALE_TYPE_LOGN,    //!< 注意力使用LogN缩放
        SCALE_TYPE_MAX      //!< 边界值，仅用于判断是否出界
    };
    //! scale类型
    //! 为SCALE_TYPE_LOGN时，不支持quanttype为2和3。
    ScaleType scaleType = SCALE_TYPE_TOR;

    //! 数据排布格式默认为BSND
    InputLayout inputLayout = TYPE_BSND;
    //! \brief 大于0时开启MLA合并kvcache功能，表示kv合并传入时v的head_size
    //! \note 默认值为0
    //! \warning 取值范围为[0,576]
    uint32_t mlaVHeadSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[68] = {0};
};

//!
//! \brief 遍历每个key和value，将key和value(num_heads, head_size)按照slotmapping填入key_cache/value_cache指定位置
//!
struct ReshapeAndCacheParam {
    //!
    //! \enum CompressType
    //!
    //! \brief 压缩类型
    //!
    //! \note 默认值为COMPRESS_TYPE_UNDEFINED(0)，不开启压缩功能。
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为非COMPRESS_TYPE_UNDEFINED(0)的值
    //!
    enum CompressType : int {
        COMPRESS_TYPE_UNDEFINED = 0, //!< 默认值，不压缩
        COMPRESS_TYPE_KVHEAD,        //!< alibi场景下压缩key_cache, value_cahe的kvHead维度
        COMPRESS_TYPE_KVHEAD_ROPE    //!< rope场景下压缩key_cache, value_cahe的kvHead维度
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置
    //!
    //! \note 默认值为K_CACHE_V_CACHE(0)，传入key_cache和value_cache
    //!
    //! \warning 仅在Atlas 800I A2推理产品上支持设置为K_CACHE_V_BYPASS(1)
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,传入key_cache和value_cache
        K_CACHE_V_BYPASS,    //!< 只传入key_cache
        K_CACHE_V_CACHE_NZ   //!< 传入key_cache和value_cache,且为NZ格式
    };

    //! 压缩方式
    CompressType compressType = COMPRESS_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvCacheCfg = K_CACHE_V_CACHE;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[16] = {0};
};

//!
//! \brief 旋转位置编码。hiddenSizeQ必须是hiddenSizeK的整数倍且满足hiddenSizeQ = headDim * headNum。
//!
struct RopeParam {
    //! \brief rope，旋转系数，对半旋转是2，支持配置2、4或headDim / 2。
    int32_t rotaryCoeff = 4;
    //! \brief 训练用参数，支持配置0或1
    int32_t cosFormat = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

//!
//! \brief 判断参数是否相同
//!
//! \param left
//! \param right
//! \return bool
//!
inline bool operator==(const RopeParam &left, const RopeParam &right)
{
    return left.rotaryCoeff == right.rotaryCoeff && left.cosFormat == right.cosFormat;
}

//!
//! \brief KVCache+KVCache+Muls+FlashAttention.
//!
struct SelfAttentionParam {
    //!
    //! \enum CalcType
    //!
    //! \brief 计算类型
    //!
    enum CalcType : int {
        UNDEFINED = 0, //!< decoder&encoder for flashAttention
        ENCODER,       //!< encoder for flashAttention
        DECODER,       //!< decoder for flashAttention
        PA_ENCODER,     //!< encoder for pagedAttention
        PREFIX_ENCODER, //!< prefix encoder for flashAttention
    };
    //!
    //! \enum KernelType
    //!
    //! \brief 算子内核精度类型
    //!
    enum KernelType : int {
        KERNELTYPE_DEFAULT = 0,   //!< i:float16, bmm:float16, o:float16
        KERNELTYPE_HIGH_PRECISION //!< i:float16, bmm:float, o:float16
    };
    //!
    //! \enum ClampType
    //!
    //! \brief clamp类型
    //!
    enum ClampType : int {
        CLAMP_TYPE_UNDEFINED = 0, //!< 不做clamp
        CLAMP_TYPE_MIN_MAX        //!< 做clamp，同时指定最大最小值
    };
    //!
    //! \enum MaskType
    //!
    //! \brief mask类型
    //!
    enum MaskType : int {
        MASK_TYPE_UNDEFINED = 0,             //!< 默认值，全0mask
        MASK_TYPE_NORM,                      //!< 倒三角mask
        MASK_TYPE_ALIBI,                     //!< alibi mask
        MASK_TYPE_NORM_COMPRESS,             //!< 倒三角压缩mask
        MASK_TYPE_ALIBI_COMPRESS,            //!< alibi压缩mask
        MASK_TYPE_ALIBI_COMPRESS_SQRT,       //!< alibi压缩开平方mask
        MASK_TYPE_ALIBI_COMPRESS_LEFT_ALIGN, //!< alibi压缩mask左对齐,只支持Atlas 800I A2推理产品
        MASK_TYPE_SLIDING_WINDOW_NORM,       //!< sliding window attention mask
        MASK_TYPE_SLIDING_WINDOW_COMPRESS    //!< sliding window attention压缩mask
    };
    //!
    //! \enum KvCacheCfg
    //!
    //! \brief KvCache配置,不支持calcType为PA_ENCODER
    //!
    enum KvCacheCfg : int {
        K_CACHE_V_CACHE = 0, //!< 默认值,进行kvcache处理
        K_BYPASS_V_BYPASS,   //!< 直接传入kvcache
    };
    //!
    //! \enum ScaleType
    //!
    //! \brief The type values of ScaleType.
    //!
    enum ScaleType : int {
        SCALE_TYPE_TOR = 0, //!< 默认值，不开启LogN缩放
        SCALE_TYPE_LOGN,    //!< 注意力使用LogN缩放，quantType只能是0
        SCALE_TYPE_MAX      //!< 边界值，仅用于判断是否出界
    };

    //! \enum QuantType
    //!
    //! \brief quant类型
    //!
    enum QuantType : int {
        TYPE_QUANT_UNDEFINED = 0,    //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_QUANT_UNQUANT = 0,      //!< 默认值，不与量化融合，此时q，k，v为bf16/float16
        TYPE_DEQUANT_FUSION = 1,     //!< 与反量化融合, 预留类型，当前不能够取此值。
        TYPE_QUANT_QKV_OFFLINE = 2,  //!< 离线INT8量化, 只支持Atlas 800I A2推理产品
        TYPE_QUANT_QKV_ONLINE = 3    //!< 在线INT8量化, 只支持Atlas 800I A2推理产品
    };
    //!
    //! \enum CacheType
    //!
    //! \brief cache内部排布类型, 为CACHE_TYPE_SWA开启SWA KVCache优化，只储存后windowSize个token的KVCache，
    //!  控制KVCache的长度不超过windowSize, 以此减少显存占用
    //!
    enum CacheType : int8_t {
        CACHE_TYPE_NORM = 0, //!< 正常cache
        CACHE_TYPE_SWA = 1   //!< 固定长度cache
    };
    //!
    //! 量化类型(只支持PA_ENCODER)：
    //! 当值为TYPE_QUANT_QKV_OFFLINE或TYPE_QUANT_QKV_ONLINE时q，k，v为int8。key,value的headsize等长，范围为（0, 256]，
    //! 且32对齐。outdatatype需要配置，只能是ACL_FLOAT16或ACL_BF16。inputLayout只支持TYPE_BSND，calcType只能为PA_ENCODER。
    QuantType quantType = TYPE_QUANT_UNQUANT;

    //! output数据类型：只支持PA_ENCODER,且QuantType不为TYPE_QUANT_UNQUANT（格式为aclDataType）
    aclDataType outDataType = ACL_DT_UNDEFINED;

    //! query头大小, 需大于0
    int32_t headNum = 0;
    //! kv头数量, 该值需要用户根据使用的模型实际情况传入
    //! kvHeadNum = 0时，keyCache的k_head_num，valueCache的v_head_num与query的num_heads一致，均为num_heads的数值
    //! kvHeadNum != 0时，keyCache的k_head_num， valueCache的v_head_num与kvHeadNum值相同
    int32_t kvHeadNum = 0;
    //! query缩放系数
    float qScale = 1;
    //! 算子tor值, 在Q*K^T后乘
    float qkScale = 1;
    //! 是否开启动态batch
    bool batchRunStatusEnable = false;
    //! 是否开启倒三角优化, 只有mask为倒三角的时候才能开启优化
    uint32_t isTriuMask = 0;
    //! 计算类型
    CalcType calcType = UNDEFINED;
    //! 内核精度类型
    KernelType kernelType = KERNELTYPE_DEFAULT;
    //! clamp类型
    ClampType clampType = CLAMP_TYPE_UNDEFINED;
    //! clamp功能最小值
    float clampMin = 0;
    //! clamp功能最大值
    float clampMax = 0;
    //! mask类型
    MaskType maskType = MASK_TYPE_UNDEFINED;
    //! kvcache配置
    KvCacheCfg kvcacheCfg = K_CACHE_V_CACHE;
    //! scale类型
    ScaleType scaleType = SCALE_TYPE_TOR;
    //! 数据排布格式默认为BSND
    InputLayout inputLayout = TYPE_BSND;
    //! \brief 大于0时开启MLA合并kvcache功能，表示kv合并传入时v的head_size
    //! \note 默认值为0
    //! \warning 取值范围为[0,576]
    uint32_t mlaVHeadSize = 0;
    //! \brief cache内部排布，开启SWA特性并设置为CACHE_TYPE_SWA可以开启SWA cache优化
    //! \note 默认值为CACHE_TYPE_NORM
    //! \warning 只有开启SWA特性后才可以是CACHE_TYPE_SWA
    CacheType cacheType = CACHE_TYPE_NORM;
    //! \brief windowSize大于0时开启SWA特性，开启SWA特性后表示sliding window 大小
    //! \note 默认值为0
    //! \warning windowSize大于0时需要将maskType设置为MASK_TYPE_SLIDING_WINDOW_NORM或MASK_TYPE_SLIDING_WINDOW_COMPRESS
    uint32_t windowSize = 0;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[64] = {0};
};

//!
//! \struct ElewiseParam
//!
//! \brief 常用的逐元素数值计算集合
//!
//! ELEWISE_ADD、ELEWISE_MUL、ELEWISE_REALDIV、ELEWISE_SUB计算类型将会对输入进行广播后再进行指定操作。
//! 输入x、y对应维度的对应值要求相同或至少其中一个为1
//!
struct ElewiseParam {
    //!
    //! \enum ElewiseType
    //!
    //! \brief 计算类型
    //!
    enum ElewiseType : int {
        ELEWISE_UNDEFINED = 0,       //!< 默认值，未定义
        ELEWISE_CAST,                //!< 数据类型转换
        ELEWISE_MULS,                //!< 向量逐元素乘值
        ELEWISE_COS,                 //!< 逐元素计算余弦值
        ELEWISE_SIN,                 //!< 逐元素计算正弦值
        ELEWISE_NEG,                 //!< 逐元素取相反数
        ELEWISE_QUANT,               //!< 量化, 仅在Atlas 800I A2推理产品上支持
        ELEWISE_LOGICAL_NOT,         //!< 逐元素逻辑非
        ELEWISE_ADD,                 //!< 逐元素相加
        ELEWISE_MUL,                 //!< 向量与向量逐元素相乘
        ELEWISE_REALDIV,             //!< 向量与向量逐元素相除
        ELEWISE_LOGICAL_AND,         //!< 逐元素逻辑与
        ELEWISE_LOGICAL_OR,          //!< 逐元素逻辑或
        ELEWISE_LESS,                //!< 逐元素判断是否小于
        ELEWISE_GREATER,             //!< 逐元素判断是否大于
        ELEWISE_SUB,                 //!< 逐元素相减
        ELEWISE_EQUAL,               //!< 逐元素判断是否相等
        ELEWISE_QUANT_PER_CHANNEL,   //!< 每个通道量化
        ELEWISE_DEQUANT_PER_CHANNEL, //!< 每个通道反量化
        ELEWISE_DYNAMIC_QUANT,       //!< 逐行动态量化
        ELEWISE_TANH,                //!< 逐元素计算双曲正切值
        ELEWISE_TYPE_MAX             //!< 边界值，仅用于判断是否出界，所有情况不能取该值
    };

    //! 量化（非每通道）所需参数
    struct QuantParam {
        //! 量化的步长
        float inputScale = 1.0f;
        //! 动态量化的是否为非对称量化
        bool asymmetric = false; //!< false : symmetric，true : asymmetric
        //! 量化的偏移度
        int inputOffset = 0;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[20] = {0};
    };

    //! 向量乘值所需参数
    struct MulsParam {
        //! 向量乘的值
        float varAttr = 0.0f;
        //!
        //! \brief 预留参数
        //!
        uint8_t rsv[12] = {0};
    };

    //! 计算方式
    ElewiseType elewiseType = ELEWISE_UNDEFINED;
    //! 量化参数
    QuantParam quantParam;
    //! 乘值参数
    MulsParam mulsParam;
    //! 指定数据类型转换输出的数据类型
    aclDataType outTensorType = ACL_DT_UNDEFINED;
    //!
    //! \brief 预留参数
    //!
    uint8_t rsv[8] = {0};
};

} // namespace infer
} // namespace atb
#endif
