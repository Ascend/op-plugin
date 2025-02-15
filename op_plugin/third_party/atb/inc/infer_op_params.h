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

} // namespace infer
} // namespace atb
#endif
