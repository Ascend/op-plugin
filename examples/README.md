# 适配说明

## 概述

C++ extensions插件提供了将自定义算子映射到昇腾AI处理器的功能，为使用PyTorch框架的开发者提供了便捷的NPU算子库调用能力，基于PyTorch原生提供的自定义算子扩展功能，用户可以编译安装自定义算子wheel包并运行。本章节基于C++ extensions的方式介绍如何在昇腾NPU上完成自定义算子开发和适配。PyTorch官方的C++ extensions功能具体可参考[C++ extensions官方文档](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

## 适配样例目录

此目录下展示了C++ extensions构建自定义算子库的所有样例。

```text
├── cpp_extension_base                 // 基础样例
├── cpp_extension_full                 // 完整样例
│   ├── module                         // 前反向绑定通过python实现
│   └── torch_library_impl             // 前反向绑定通过c++实现
├── cpp_extension_structured           // 结构化适配实现
├── cpp_extension_asc                  // 算子通过AscendC实现
└── cpp_extension_pybind               // 通过pybind绑定对外接口
```

## 适配方法

提供了六种适配样例以供用户选择，用户可根据需要选择对应适配方法。

| 适配方法                                                        | 使用场景|功能范围 |                                                更多信息                                                 |
| ------------------------------------------------------------ | :------: |  :------:  |:---------------------------------------------------------------------------------------------------:|
|cpp_extension_base（基础样例）       |基础功能| 算子定义、算子适配、ATen IR注册绑定|                          具体操作请参考[基础调用适配样例](./cpp_extension_base/README.md)                          |
|cpp_extension_full（完整样例-module）  |完整功能 | 算子定义、算子适配、ATen IR注册绑定、meta注册、前反向绑定等 | 前反向绑定通过Python类的`forward`和`backward`注册实现<br>具体操作请参考[完整调用适配样例](./cpp_extension_full/module/README.md) |
|cpp_extension_full（完整样例-`TORCH_LIBRARY_IMPL`）  |完整功能 | 算子定义、算子适配、ATen IR注册绑定、meta注册、前反向绑定等 | 前反向绑定通过`TORCH_LIBRARY_IMPL`注册实现<br>具体操作请参考[完整调用适配样例](./cpp_extension_full/torch_lib_impl/README.md) |
|cpp_extension_structured（结构化）  |结构化适配|  算子定义、算子适配、ATen IR注册绑定|                通过配置yaml自动完成适配，具体操作请参考[结构化适配样例](./cpp_extension_structured/README.md)                |
|cpp_extension_asc（AscendC）  | 自定义算子   | 算子定义、算子适配、ATen IR注册绑定 |                 算子使用AscendC编写，具体操作请参考[AscendC算子适配样例](./cpp_extension_asc/README.md)                 |
|cpp_extension_pybind（pybind）  | 灵活接口  | 算子定义、算子适配、Pybind绑定 |                       具体操作请参考[pybind绑定适配样例](./cpp_extension_pybind/README.md)                       |

> [!NOTE]
> 
> cpp_extension_structured（结构化实现）使用前提：opapi对应的aclnn与ATen IR语义一致，适配层除申请output tensor外无其他逻辑。
