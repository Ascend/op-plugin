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

#ifndef INC_EXTERNAL_ATB_TYPES_H
#define INC_EXTERNAL_ATB_TYPES_H
#include <cstdint>
#include <functional>
#include <vector>
#include <string>
#include <acl/acl.h>
#include "./svector.h"

//!
//! \file types.h
//!
//! \brief 定义加速库各种数据类型及日志错误类型
//!

namespace atb {
//! \brief 用一系列状态值表示加速库中的返回值
using Status = int32_t;
//! \brief 数据最大维度定义
constexpr uint32_t MAX_DIM = 8;

//!
//! \enum ErrorType
//!
//! \brief 加速库日志中定义的错误类型
//!
enum ErrorType : int {
    NO_ERROR = 0,                         //!< 正确
    ERROR_INVALID_PARAM,                  //!< 无效参数
    ERROR_INVALID_GRAPH,                  //!< 计算图错误
    ERROR_INTERNAL_ERROR,                 //!< 内部错误
    ERROR_RT_FAIL,                        //!< 调用Runtime接口失败
    ERROR_INVALID_IN_TENSOR_NUM,          //!< 算子输入Tensor数量与定义不一致
    ERROR_INVALID_TENSOR_DTYPE,           //!< Tensor数据类型错误
    ERROR_INVALID_TENSOR_FORMAT,          //!< Tensor数据格式错误
    ERROR_INVALID_TENSOR_DIM,             //!< Tensor数据维度错误
    ERROR_INVALID_TENSOR_SIZE,            //!< Tensor数据size错误
    ERROR_OPERATION_NULL_RUNNER,          //!< Operation内部错误
    ERROR_GRAPH_INFERSHAPE_FUNC_FAIL,     //!< 图infershapeFunc错误
    ERROR_CANN_ERROR,                     //!< 调用CANN接口错误
    ERROR_INVALID_TENSOR_INI_MATCH,       //!< ini配置文件校验失败
    ERROR_INVALID_TENSOR_ADDR,            //!< Tensor地址错误
    ERROR_INVALID_TENSOR_NUM,             //!< Tensor数量错误
    ERROR_INVALID_TENSOR_DIM_NUM,         //!< Tensor维度数错误
    ERROR_INVALID_SINGLE_OPERATION_PARAM, //!< 单Operation参数错误
    ERROR_GRAPH_NODE_RESHAPE_FUNC_FAIL,   //!< 图节点reshapeFuncs设置有误
    ERROR_INVALID_GRAPH_NODE_CHUNK,       //!< 图节点Chunk参数错误
    ERROR_INVALID_CONTEXT_ADDR,           //!< Context地址错误
    ERROR_INVALID_STREAM,                 //!< Stream错误
    ERROR_INVALID_WORKSPACE_SIZE,         //!< Workspace大小有误
    ERROR_INVALID_WORKSPACE_ADDR,         //!< Workspace地址有误
    ERROR_INVALID_OPERATION_ADDR,         //!< Operation地址有误
    ERROR_HCCL_FAIL,                      //!< HCCL接口调用失败
    ERROR_OUT_OF_DEVICE_MEMORY,           //!< Device内存不足
    ERROR_OUT_OF_HOST_MEMORY              //!< Host内存不足
};
//!
//! \struct Dims
//!
//! \brief Shape维度信息。
//!
struct Dims {
    //! \brief 每一维的大小，要求大于0。
    int64_t dims[MAX_DIM];
    //! \brief Tensor的维数，取值范围为(0, 8]。
    uint64_t dimNum = 0;
};

//!
//! \struct TensorDesc
//!
//! \brief 包含对Tensor的相关描述信息：每个Tensor的数据类型，数据排布格式和形状维度信息。
//!
//! \warning Atlas 推理系列产品 中不支持ACL_BF16（bf16）类型数据。
//!
struct TensorDesc {
    //! \brief Tensor数据类型
    aclDataType dtype = ACL_DT_UNDEFINED;
    //! \brief Tensor数据排布格式
    aclFormat format = ACL_FORMAT_UNDEFINED;
    //! \brief Tensor数据的形状
    Dims shape;
};

//!
//! \struct Tensor
//!
//! \brief 加速库的Tensor定义，包含每个Tensor的描述信息、NPU内存地址、CPU内存地址和内存大小等。
//!
//!
struct Tensor {
    //! \brief Tensor描述信息
    TensorDesc desc;
    //! \brief TensorNPU内存地址。
    void *deviceData = nullptr;
    //! \brief TensorCPU内存地址。
    void *hostData = nullptr;
    //! \brief “deviceData”或“hostData”指向内容的内存大小。
    uint64_t dataSize = 0;
};

//!
//! \struct VariantPack
//!
//! \brief 加速库算子执行时需要构造VariantPack存放输入及最终输出
//!
//! \see Tensor,SVector
//!
struct VariantPack {
    //! \brief 存放所有输入Tensor的SVector
    SVector<Tensor> inTensors;
    //! \brief 存放所有输出Tensor的SVector
    SVector<Tensor> outTensors;
};

//!
//! \struct Chunk
//!
//! \brief 在host侧对Tensor执行split切分操作
//!
struct Chunk {
    //! \brief 切分数量
    uint32_t chunkNum = 1;
    //! \brief 标记均分后使用是的第几份数据
    uint32_t chunkIndex = 0;
};

//! \brief Reshape功能，改变Tensor的shape。
using ReshapeFunc = std::function<void(const Dims &oldShape, Dims &newShape)>;
//! \brief 根据输入inTensor信息推导输出outTensor信息。
using InferShapeFunc =
    std::function<Status(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs)>;

class Operation;

//!
//! \struct Node
//!
//! \brief 图算子中的Operation节点，每个Node表示一个Operation或者GraphOperation，所有的Node组成一个完整的图算子。
//!
//! \note inTensorIds、outTensorIds和inTensorReshapeFuncs均为SVector，每个元素的顺序需要和对应Tensor的顺序保持一致。
//!
struct Node {
    //! \brief Node对应的operation或者graphOperation。
    Operation *operation = nullptr;
    //! \brief Node对应的operation或者graphOperation的输入tensorId SVector。
    SVector<uint32_t> inTensorIds;
    //! \brief Node对应的operation或者graphOperation的输出tensorId SVector。
    SVector<uint32_t> outTensorIds;
    //! \brief Node对应的operation或者graphOperation的每个输入Tensor的reshape函数SVector。
    SVector<ReshapeFunc> inTensorReshapeFuncs;
    //! \brief 存放chunk
    SVector<Chunk> inTensorChunks;
};

//!
//! \struct GraphParam
//!
//! \brief 图算子参数。
//!
struct GraphParam {
    //! \brief 图名称。仅允许字母、数字、下划线，名称长度不超过128。
    std::string name;
    //! \brief 图算子输入Tensor的数量，需小于或等于256
    uint32_t inTensorNum = 0;
    //! \brief 图算子输出Tensor的数量，需小于或等于256
    uint32_t outTensorNum = 0;
    //! \brief 图算子中间Tensor的数量，需小于或等于256
    uint32_t internalTensorNum = 0;
    //!
    //! \brief 图算子Node Vector。
    //!
    //! nodes的长度满足小于1024。
    //! nodes的顺序，需要满足各个node对应operation的执行顺序依赖关系，先执行的在前面，后执行的在后面。
    //! 如果inTensorNum，outTensorNum与internalTensorNum之和为S，则nodes中各个元素的inTensorIds，outTensorIds中各元素值均要求落在[0, S-1]范围内。
    //!
    std::vector<Node> nodes;
    //! \brief inferShape函数指针。
    InferShapeFunc inferShapeFunc = nullptr;
};
} // namespace atb
#endif
