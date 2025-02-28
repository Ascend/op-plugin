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

#ifndef INC_EXTERNAL_ATB_OPERATION_H
#define INC_EXTERNAL_ATB_OPERATION_H
#include <cstdint>
#include <functional>
#include <string>
#include "./types.h"
#include "./svector.h"
#include "./context.h"

//!
//! \file operation.h
//!
//! \brief 定义加速库Operation类
//!

namespace atb {

//!
//! \class Operation.
//!
//! \brief 加速库Operation类.
//!
//! 该接口类定义了算子准备与执行的需要的一系列的接口，通过创建Operation可以执行算子
//!
class Operation {
public:
    //! \brief 默认构造函数.
    Operation() = default;

    //! \brief 默认析构函数.
    virtual ~Operation() = default;
    //!
    //! \brief 获取创建的Operation的名字
    //!
    //! \return 返回字符串
    //!
    virtual std::string GetName() const = 0;

    //!
    //! \brief 根据输入Tensor描述信息推导出输出Tensor的描述信息。
    //!
    //! \param inTensorDescs 存放所有输入tensor描述信息的SVector
    //! \param outTensorDescs 存放所有输出tensor描述信息的SVector
    //!
    //! \return 状态值，如果成功，返回NO_ERROR
    //!
    virtual Status InferShape(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const = 0;

    //!
    //! \brief 获取Op/GraphOp输入Tensor个数接口。
    //!
    //! \return 整数值
    //!
    virtual uint32_t GetInputNum() const = 0;

    //!
    //! \brief 获取Op/GraphOp输出Tensor个数接口。
    //!
    //! \return 整数值
    //!
    virtual uint32_t GetOutputNum() const = 0;

    //!
    //! \brief Operation执行前的一系列准备工作
    //!
    //! 主要是计算Operation执行过程需要分配的内存空间workspaceSize
    //!
    //! \param variantPack 输入与输出Tensor
    //! \param workspaceSize 获取Operation执行需要分配的内存空间
    //! \param context Operation执行准备工作所在的上下文
    //!
    //! \return 状态值，如果成功，返回NO_ERROR
    //!
    virtual Status Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context) = 0;

    //!
    //! \brief Operation执行的流程
    //!
    //! 根据setup过程中得到的workspaceSize为Operation执行分配实际的内存，并执行Operation
    //!
    //! \param variantPack 输入与输出Tensor
    //! \param workspace Operation执行分配的内存地址
    //! \param workspaceSize Operation执行需要分配的内存空间
    //! \param context Operation执行所在的上下文
    //!
    //! \return 状态值，如果成功，返回NO_ERROR
    //!
    virtual Status Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                           Context *context) = 0;
};

//!
//! \brief 创建Operation
//!
//! \param opParam 根据参数来指定调用的Operation
//! \param operation Operation指针地址
//!
//! \return 状态值，如果成功，返回NO_ERROR
//!
template <typename OpParam> Status CreateOperation(const OpParam &opParam, Operation **operation);

//!
//! \brief 销毁Operation
//!
//! \param operation Operation指针
//!
//! \return 状态值，如果成功，返回NO_ERROR
//!
//! \note 调用CreateOperation接口创建Operation，执行完Operation后需要调用DestroyOperation接口进行销毁。否则将导致内存泄漏。
//!
Status DestroyOperation(Operation *operation);

//!
//! \brief 拷贝Operation的Param参数
//!
//! \param operation Operation指针
//! \param opParam OpParam的引用，将返回operation的opParam浅拷贝
//!
//! \return 状态值，如果成功，返回NO_ERROR
//!
template <typename OpParam> Status CloneOperationParam(const Operation *operation, OpParam &opParam);

//!
//! \brief 更新Operation的Param参数
//!
//! \param operation Operation指针
//! \param opParam Operation新的param值
//!
//! \return 状态值，如果成功，返回NO_ERROR
//!
template <typename OpParam> Status UpdateOperationParam(Operation *operation, const OpParam &opParam);

} // namespace atb
#endif
