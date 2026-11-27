# 适配说明

## 概述

本章节通过样例介绍如何使用PyTorch的C++扩展机制开发和适配昇腾NPU自定义算子，以及如何编译、安装和运行扩展wheel包。PyTorch的C++扩展机制请参见[官方文档](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

## 适配样例目录

此目录下展示了C++ extensions构建自定义算子库的所有样例。

```text
├── cpp_extension_base                 // 基础样例
├── cpp_extension_full                 // 完整样例
│   ├── module                         // 前反向绑定通过python实现
│   └── torch_lib_impl             // 前反向绑定通过c++实现
├── cpp_extension_structured           // 结构化适配实现
├── cpp_extension_asc                  // 算子通过Ascend C实现
└── cpp_extension_pybind               // 通过pybind11提供Python调用接口
```

## 适配方法

提供了六种适配样例以供用户选择，用户可根据需要选择对应适配方法。

| 适配方法                                                        | 使用场景|功能范围 |                                                更多信息                                                 |
| ------------------------------------------------------------ | :------: |  :------:  |:---------------------------------------------------------------------------------------------------:|
|cpp_extension_base（基础样例）       |基础功能| 算子定义、算子适配、ATen IR注册绑定|                          具体操作请参考[基础调用适配样例](./cpp_extension_base/README.md)                          |
|cpp_extension_full（完整样例-module）  |完整功能 | 算子定义、算子适配、ATen IR注册绑定、meta注册、前反向绑定等 | 前反向绑定通过Python类的`forward`和`backward`注册实现<br>具体操作请参考[完整调用适配样例](./cpp_extension_full/module/README.md) |
|cpp_extension_full（完整样例-`TORCH_LIBRARY_IMPL`）  |完整功能 | 算子定义、算子适配、ATen IR注册绑定、meta注册、前反向绑定等 | 前反向绑定通过`TORCH_LIBRARY_IMPL`注册实现<br>具体操作请参考[完整调用适配样例](./cpp_extension_full/torch_lib_impl/README.md) |
|cpp_extension_structured（结构化）  |结构化适配|  算子定义、算子适配、ATen IR注册绑定|                通过配置yaml自动完成适配，具体操作请参考[结构化适配样例](./cpp_extension_structured/README.md)                |
|cpp_extension_asc（Ascend C）  | 自定义算子   | 算子定义、算子适配、ATen IR注册绑定 |                 算子使用Ascend C编写，具体操作请参考[Ascend C算子适配样例](./cpp_extension_asc/README.md)                 |
|cpp_extension_pybind（pybind11绑定）  | 灵活接口  | 算子定义、算子适配、pybind11绑定 |                       具体操作请参考[pybind11绑定适配样例](./cpp_extension_pybind/README.md)                       |

> [!NOTE]
> 
> cpp_extension_structured（结构化实现）使用前提：opapi对应的aclnn与ATen IR语义一致，适配层除申请output tensor外无其他逻辑。
