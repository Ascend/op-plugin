/**
 * @file registration.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"

// 在myops命名空间里注册add_custom和add_custom_backward两个schema，新增自定义aten ir需要在此注册
TORCH_LIBRARY(myops, m) {
    m.def("add_custom(Tensor self, Tensor other) -> Tensor");
    m.def("add_custom_backward(Tensor self) -> (Tensor, Tensor)");
}

// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_custom", &add_custom_autograd, "x + y");
}
