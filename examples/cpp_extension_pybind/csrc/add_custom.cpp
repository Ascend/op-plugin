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
    const c10::OptionalDeviceGuard device_guard(device_of(self));
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    at::Scalar alpha = 1.0;

    // 调用aclnn接口计算
    EXEC_NPU_CMD_EXT(aclnnAdd, self, other, alpha, result);
    return result;
}

// expose Ascend custom ops to Python
PYBIND11_MODULE(custom_ops_lib, m)
{
    m.def("add_custom", &add_custom_impl_npu, "");
}

