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

#ifndef INC_EXTERNAL_ATB_SVECTOR_H
#define INC_EXTERNAL_ATB_SVECTOR_H
#include <vector>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <initializer_list>

//!
//! \file svector.h
//!
//! \brief 定义加速库容器类
//!

namespace atb {
//! \brief 最大容器大小
constexpr size_t MAX_SVECTOR_SIZE = 256;
//! \brief 默认容器大小
constexpr size_t DEFAULT_SVECTOR_SIZE = 64;
//! \brief 边界检查标志
constexpr bool CHECK_BOUND = true;

struct MaxSizeExceeded : public std::exception {};

//!
//! \class SVector
//!
//! \brief 加速库svector类
//!
//! 封装动态数组的顺序容器
//!
template <class T> class SVector {
public:
    //! \brief 默认构造函数
    //!
    //! \note 容量为DEFAULT_SVECTOR_SIZE
    //!
    constexpr SVector() : size_(0)
    {
        for (std::size_t i = 0; i < DEFAULT_SVECTOR_SIZE; ++i) {
            storage_[i] = {};
        }
    }
    //! \brief 初始化列表构造函数
    //!
    //! \param list
    //!
    //! \note list长度需小于DEFAULT_SVECTOR_SIZE，否则会抛出异常
    //!
    SVector(std::initializer_list<T> list)
    {
        if (CHECK_BOUND && list.size() > DEFAULT_SVECTOR_SIZE) {
            throw MaxSizeExceeded();
        }
        size_ = list.size();
        size_t i = 0;
        for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
            storage_[i++] = *it;
        }
    }
    //! \brief 带参数的构造函数
    //!
    //! \param size
    //! \param value
    //!
    //! \note size大小需小于DEFAULT_SVECTOR_SIZE，否则会抛出异常
    //!
    explicit SVector(std::size_t size, const T &value = 0) : size_(0)
    {
        if (CHECK_BOUND && size > DEFAULT_SVECTOR_SIZE) {
            throw MaxSizeExceeded();
        }
        size_ = size;
        for (std::size_t i = 0; i < size_; ++i) {
            storage_[i] = value;
        }
    }
    //! \brief 拷贝构造函数，创建一个新的SVector对象并将另一个SVector对象的值复制到新对象
    //!
    //! \param other
    //!
    SVector(const SVector<T> &other)
    {
        if (other.heap_) {
            heap_ = reinterpret_cast<T *>(malloc(other.size_ * sizeof(T)));
            if (!heap_) {
                throw std::bad_alloc();
            }
            size_ = other.size_;
            for (std::size_t i = 0; i < other.size_; ++i) {
                heap_[i] = other.heap_[i];
            }
        } else {
            size_ = other.size_;
            for (std::size_t i = 0; i < other.size_; ++i) {
                storage_[i] = other.storage_[i];
            }
        }
    }

    ~SVector()
    {
        if (heap_) {
            free(heap_);
        }
    }

    //! \brief 插入元素到指定容器
    //!
    //! \param val
    //!
    //! \note 待添加SVector内元素必须小于SVector容量，否则会抛出异常
    //!
    void push_back(const T &val) noexcept((!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (heap_) {
            if (CHECK_BOUND && size_ == capacity_) {
                throw MaxSizeExceeded();
            }
            heap_[size_++] = val;
            return;
        }
        if (CHECK_BOUND && size_ == DEFAULT_SVECTOR_SIZE) {
            throw MaxSizeExceeded();
        }
        storage_[size_++] = val;
    }

    //! \brief 获取容器起始元素地址
    //!
    //! \return 指针
    //!
    T *begin() noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    //! \brief 获取容器起始元素地址
    //!
    //! \return 常量指针
    //!
    const T *begin() const noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    //! \brief 获取容器尾元素地址
    //!
    //! \return 指针
    //!
    T *end() noexcept
    {
        if (heap_) {
            return (&heap_[0]) + size_;
        }
        return (&storage_[0]) + size_;
    }

    //! \brief 获取容器尾地址
    //!
    //! \return 常量指针
    //!
    const T *end() const noexcept
    {
        if (heap_) {
            return (&heap_[0]) + size_;
        }
        return (&storage_[0]) + size_;
    }

    //! \brief 访问指定位置的元素
    //!
    //! \param i
    //!
    //! \return 引用
    //!
    T &operator[](std::size_t i)
    {
        if (heap_) {
            if (size_ == 0 || i >= size_) {
                throw std::out_of_range("out of range");
            }
            return heap_[i];
        }
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    //! \brief 访问指定位置的元素
    //!
    //! \param i
    //!
    //! \return 常量引用
    //!
    const T &operator[](std::size_t i) const
    {
        if (heap_) {
            if (size_ == 0 || i >= size_) {
                throw std::out_of_range("out of range");
            }
            return heap_[i];
        }
        if (size_ == 0 || i >= size_) {
            throw std::out_of_range("out of range");
        }
        return storage_[i];
    }

    //! \brief 访问指定位置的元素
    //!
    //! \param i
    //!
    //! \return 引用
    //!
    T &at(std::size_t i)
    {
        if (heap_) {
            if (size_ == 0 || i >= size_) {
                throw std::out_of_range("out of range");
            }
            return heap_[i];
        } else {
            if (size_ == 0 || i >= size_ || i > DEFAULT_SVECTOR_SIZE) {
                throw std::out_of_range("out of range");
            }
            return storage_[i];
        }
    }

    //! \brief 访问指定位置的元素
    //!
    //! \param i
    //!
    //! \return 引用
    //!
    const T &at(std::size_t i) const
    {
        if (heap_) {
            if (size_ == 0 || i >= size_) {
                throw std::out_of_range("heap out of range");
            }
            return heap_[i];
        }
        if (size_ == 0 || i >= size_ || i > DEFAULT_SVECTOR_SIZE) {
            throw std::out_of_range("stack out of range");
        }
        return storage_[i];
    }

    //! \brief 获取容器的大小
    //!
    //! \return size
    //!
    std::size_t size() const noexcept
    {
        return size_;
    }

    //! \brief 向容器内指定位置插入元素
    //!
    //! \param pos
    //! \param value
    //!
    //! \note pos必须小于SVector容量，否则会抛出异常
    //!
    void insert(const std::size_t pos,
                const T &value) noexcept((!CHECK_BOUND) && std::is_nothrow_assignable<T, const T &>::value)
    {
        if (heap_) {
            if (pos > size_ || pos == capacity_) {
                throw MaxSizeExceeded();
            }
            for (auto it = size_; it != pos; it--) {
                heap_[it] = heap_[it - 1];
            }
            heap_[pos] = value;
            size_ += 1;
            return;
        }
        if (CHECK_BOUND && size_ == DEFAULT_SVECTOR_SIZE) {
            throw MaxSizeExceeded();
        }
        if (pos > size_) {
            throw MaxSizeExceeded();
        }

        for (auto it = size_; it != pos; it--) {
            storage_[it] = storage_[it - 1];
        }
        storage_[pos] = value;
        size_ += 1;
        return;
    }

    //! \brief 判断容器是否为空
    //!
    //! \return bool值
    //!
    bool empty() const noexcept
    {
        return size_ == 0;
    }

    //! \brief 清空容器
    void clear() noexcept
    {
        size_ = 0;
    }

    //! \brief 获取容器起始元素地址
    //!
    //! \return 指针
    //!
    T *data() noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    //! \brief 获取容器起始元素地址
    //!
    //! \return 常量指针
    //!
    const T *data() const noexcept
    {
        if (heap_) {
            return &heap_[0];
        }
        return &storage_[0];
    }

    //! \brief 改变SVector容器大小，不能改变SVector容量
    //!
    //! \param size
    //!
    //! \note 传入size参数不能超过SVector容量，反之，则会抛出异常。
    //!
    void resize(std::size_t size)
    {
        if (heap_ && size > capacity_) {
            throw MaxSizeExceeded();
        }
        size_ = size;
    }

    //! \brief 改变SVector容量大小，清空内部数据，并将SVector容量大小定义为size大小
    //!
    //! \param size
    //!
    //! \note 用于预分配内存空间，SVector默认容量为DEFAULT_SVECTOR_SIZE，传入的size需大于DEFAULT_SVECTOR_SIZ且小于MAX_SVECTOR_SIZE，反之，则会抛出异常。
    //!
    void reserve(std::size_t size)
    {
        if (size > MAX_SVECTOR_SIZE) {
            throw MaxSizeExceeded();
        }

        if (size > DEFAULT_SVECTOR_SIZE) {
            if (heap_) {
                free(heap_);
            }
            heap_ = reinterpret_cast<T *>(malloc(size * sizeof(T)));
            if (!heap_) {
                throw std::bad_alloc();
            }
            for (std::size_t i = 0; i < size; ++i) {
                heap_[i] = {};
            }
            capacity_ = size;
        }
    }

    //! \brief 判断两个容器中的元素是否全部相同
    //!
    //! \param other
    //!
    //! \return bool值
    //!
    bool operator==(const SVector<T> &other) const
    {
        if (heap_) {
            if (size_ != other.size_ || !other.heap_) {
                return false;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (heap_[i] != other.heap_[i]) {
                    return false;
                }
            }
        } else {
            if (size_ != other.size_) {
                return false;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (storage_[i] != other.storage_[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    //! \brief 判断两个容器中的元素是否存在不同
    //!
    //! \param other
    //!
    //! \return bool值
    //!
    bool operator!=(const SVector<T> &other) const
    {
        if (heap_) {
            if (size_ != other.size_ || !other.heap_) {
                return true;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (heap_[i] != other.heap_[i]) {
                    return true;
                }
            }
        } else {
            if (size_ != other.size_) {
                return true;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (storage_[i] != other.storage_[i]) {
                    return true;
                }
            }
        }
        return false;
    }

    //! \brief 判断一个容器中的元素是否比另一个容器小
    //!
    //! \param other
    //!
    //! \return bool值
    //!
    bool operator<(const SVector<T> &other) const
    {
        if (heap_) {
            if (size_ != other.size_ || !other.heap_) {
                return size_ < other.size_;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (heap_[i] != other.heap_[i]) {
                    return heap_[i] < other.heap_[i];
                }
            }
        } else {
            if (size_ != other.size_) {
                return size_ < other.size_;
            }
            for (size_t i = 0; i < size_; ++i) {
                if (storage_[i] != other.storage_[i]) {
                    return storage_[i] < other.storage_[i];
                }
            }
        }
        return false;
    }

    //! \brief 重载运算符函数，将初始化列表中的元素赋值给一个SVector对象
    //!
    //! \param list
    //!
    //! \return 容器引用
    //!
    SVector &operator=(std::initializer_list<T> list)
    {
        if (heap_) {
            if (CHECK_BOUND && list.size() > MAX_SVECTOR_SIZE) {
                throw MaxSizeExceeded();
            }
            size_ = list.size();
            size_t i = 0;
            for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
                heap_[i++] = *it;
            }
            return *this;
        } else {
            if (CHECK_BOUND && list.size() > DEFAULT_SVECTOR_SIZE) {
                throw MaxSizeExceeded();
            }
            size_ = list.size();
            size_t i = 0;
            for (auto it = list.begin(); it != list.end() && i < size_; ++it) {
                storage_[i++] = *it;
            }
            return *this;
        }
    }

    //! \brief 用一个容器给另一个容器赋值
    //!
    //! \param other
    //!
    //! \return 容器引用
    //!
    SVector &operator=(const SVector &other)
    {
        if (heap_) {
            size_ = other.size_;
            for (std::size_t i = 0; i < other.size_; ++i) {
                heap_[i] = other.heap_[i];
            }
            return *this;
        } else {
            size_ = other.size_;
            for (std::size_t i = 0; i < other.size_; ++i) {
                storage_[i] = other.storage_[i];
            }
            return *this;
        }
    }

private:
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
    T storage_[DEFAULT_SVECTOR_SIZE + 1];
    T *heap_ = nullptr;
};

//! \brief 输出容器中的元素
//!
//! \param os
//! \param svector
//!
//! \return 输出流
//!
template <class T> std::ostream &operator<<(std::ostream &os, const SVector<T> &svector)
{
    if (svector.size() == 0) {
        return os;
    }

    std::string str = ",";
    for (size_t i = 0; i < svector.size(); ++i) {
        os << svector.at(i);
        if (i != svector.size() - 1) {
            os << str;
        }
    }

    return os;
}
} // namespace atb
#endif