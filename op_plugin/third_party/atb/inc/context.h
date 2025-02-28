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

#ifndef INC_EXTERNAL_ATB_CONTEXT_H
#define INC_EXTERNAL_ATB_CONTEXT_H
#include <acl/acl.h>
#include "./types.h"

//!
//! \file context.h
//!
//! \brief 定义加速库上下文类
//!

//!
//! \namespace atb
//!
//! \brief 加速库的命名空间.
//!
namespace atb {

//!
//! \class Context.
//!
//! \brief 加速库上下文类，主要用于管理Operation运行所需要的全局资源.
//!
//! Context类会管理任务流队列比如Operation执行以及TilingCopy,管理tiling内存的申请与释放.
//!
class Context {
public:
    //! \brief 默认构造函数.
    Context() = default;

    //! \brief 默认析构函数.
    virtual ~Context() = default;

    //!
    //! \brief 将传入stream队列设置为当前执行队列.
    //!
    //! 将传入stream队列设置为当前执行队列,然后再去执行对应的Operation.
    //!
    //! \param stream 传入的stream队列
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetExecuteStream(aclrtStream stream) = 0;

    //!
    //! \brief 获取当前执行stream队列.
    //!
    //! \return 执行流队列
    //!
    virtual aclrtStream GetExecuteStream() const = 0;

    //!
    //! \brief 设置异步拷贝tiling信息功能.
    //!
    //! 设置异步拷贝tiling信息功能是否开启，如果是，则创建stream和event来进行tiling拷贝过程.
    //!
    //! \param enable 传入的标志，bool类型
    //!
    //! \return 状态值.如果设置成功，返回NO_ERROR.
    //!
    virtual Status SetAsyncTilingCopyStatus(bool enable) = 0;

    //!
    //! \brief 获取tiling拷贝状态.
    //!
    //! \return 如果获取成功，返回True.
    //!
    virtual bool GetAsyncTilingCopyStatus() const = 0;
};

//!
//! \brief 创建上下文.
//!
//! 在当前进程或线程中显式创建一个Context.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status CreateContext(Context **context);

//!
//! \brief 销毁上下文.
//!
//! 销毁上下文中所有的资源.
//!
//! \param context 传入的context
//!
//! \return 状态值.如果设置成功，返回NO_ERROR.
//!
Status DestroyContext(Context *context);
} // namespace atb
#endif
