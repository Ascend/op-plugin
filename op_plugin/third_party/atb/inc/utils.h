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

#ifndef INC_EXTERNAL_ATB_UTILS_H
#define INC_EXTERNAL_ATB_UTILS_H
#include <cstdint>
#include "./types.h"

//!
//! \file utils.h
//!
//! \brief 定义加速库公共数据接口类
//!

namespace atb {

//!
//! \class Utils.
//!
//! \brief 加速库公共工具接口类.
//!
//! 该接口类定义了一系列的公共接口
//!
class Utils {
public:
    //!
    //! \brief 获取加速库版本信息。
    //!
    //! \return 返回字符串类型.
    //!
    static std::string GetAtbVersion();

    //!
    //! \brief 返回Tensor对象的数据存储大小。
    //!
    //! \param tensor 传入Tensor
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorSize(const Tensor &tensor);

    //!
    //! \brief 返回Tensor对象的数据存储大小。
    //!
    //! \param tensorDesc 传入TensorDesc
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorSize(const TensorDesc &tensorDesc);

    //!
    //! \brief 返回Tensor对象的数据个数。
    //!
    //! \param tensor 传入Tensor
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorNumel(const Tensor &tensor);

    //!
    //! \brief 返回Tensor对象的数据个数。
    //!
    //! \param tensorDesc 传入TensorDesc
    //!
    //! \return 返回整数值
    //!
    static uint64_t GetTensorNumel(const TensorDesc &tensorDesc);

    //!
    //! \brief 量化场景使用。float数组转成uint64数组，实现逻辑是复制float到uint64的后32位，uint64的前32位置0。
    //!
    //! \param src 输入float数组
    //! \param dest 转化得到的uint64数组
    //! \param itemCount 数组元素个数
    //!
    static void QuantParamConvert(const float *src, uint64_t *dest, uint64_t itemCount);
};
} // namespace atb
#endif
