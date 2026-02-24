/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>
#include "npu_cpp_extension.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

// 为NPU设备注册前向实现
at::Tensor add_custom_impl_npu(const at::Tensor& self, const at::Tensor& other)
{
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    at::Scalar alpha = 1.0;

    // 调用aclnn接口计算
    EXEC_NPU_CMD_EXT(aclnnAdd, self, other, alpha, result);
    return result;
}

// 为NPU设备注册反向实现
std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_npu(const at::Tensor& grad)
{
    at::Tensor result = grad; // 创建输出内存

    return {result, result};
}

// 为Meta设备注册前向实现
at::Tensor add_custom_impl_meta(const at::Tensor& self, const at::Tensor& other)
{
    return at::empty_like(self);
}

// 为Meta设备注册反向实现
std::tuple<at::Tensor, at::Tensor> add_custom_backward_impl_meta(const at::Tensor& self)
{
    auto result = at::empty_like(self);
    return std::make_tuple(result, result);
}

// 通过继承torch::autograd::Function类实现前反向绑定
class AddCustomFunction : public torch::autograd::Function<AddCustomFunction> {
public:
    static at::Tensor forward(Autog radContext *ctx, at::Tensor self, at::Tensor other)
    {
        at::AutoDispatchBelowADInplaceOrView guard;
        static auto op = torch::Dispatcher::singleton()
                        .findSchemaOrThrow("myops::add_custom", "")
                        .typed<decltype(add_custom_impl_npu)>();

        auto result = op.call(self, other);
        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];

        static auto op = torch::Dispatcher::singleton()
                                        .findSchemaOrThrow("myops::add_custom_backward", "")
                        .typed<decltype(add_custom_backward_impl_npu)>();
        auto result = op.call(grad_output);
        return {std::get<0>(result), std::get<1>(result)};
    }
};

// 使用的时候调用apply()方法
at::Tensor add_custom_autograd(const at::Tensor& self, const at::Tensor& other)
{
    return AddCustomFunction::apply(self, other);
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_custom", &add_custom_impl_npu);
    m.impl("add_custom_backward", &add_custom_backward_impl_npu);
}

// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
    m.impl("add_custom", &add_custom_autograd);
}

// 为Meta设备注册前反向实现
TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("add_custom", &add_custom_impl_meta);
    m.impl("add_custom_backward", &add_custom_backward_impl_meta);
}
