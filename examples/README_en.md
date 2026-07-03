# Adaptation Description

## Overview

The C++ extensions plugin maps custom operators to Ascend AI Processors, allowing PyTorch developers to seamlessly access the NPU operator library. Based on the native custom operator extension capabilities provided by PyTorch, developers can compile, install, and execute the custom operator wheel package. This chapter describes how to develop and adapt custom operators on Ascend NPUs using C++ extensions. For details about the official PyTorch C++ extensions feature, see [PyTorch C++ Extensions documentation](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

## Directory Structure

This directory contains all example projects for building custom operator libraries by using C++ extensions.

```text
├── cpp_extension_base                 // Basic example
├── cpp_extension_full                 // Full example
│   ├── module                         // Forward and backward bindings implemented in Python
│   └── torch_library_impl             // Forward and backward bindings implemented in C++
├── cpp_extension_structured           // Structured adaptation
├── cpp_extension_asc                  // Ascend C-based operator implementation
└── cpp_extension_pybind               // Interface binding implemented using pybind
```

## Adaptation Methods

Six adaptation example projects are provided. You can choose the appropriate adaptation method as needed.

| Adaptation Method                                                       | Scenario|Supported Features|                                                More Information                                                |
| ------------------------------------------------------------ | :------: |  :------:  |:---------------------------------------------------------------------------------------------------:|
|`cpp_extension_base` (Basic Example)      |Basic features| Operator definition, operator adaptation, and ATen IR registration and binding|                          For details, see [Adaptation Development and Usage (Basic Example)](./cpp_extension_base/README.md).                         |
|`cpp_extension_full` (Complete Module Example) |Full features| Operator definition, operator adaptation, ATen IR registration and binding, meta function registration, and forward/backward bindings| Forward and backward bindings are implemented using the Python class `forward` and `backward` methods.<br>For details, see the [Adaptation Development and Usage (Complete Module Example)](./cpp_extension_full/module/README.md).|
|`cpp_extension_full` (Complete `TORCH_LIBRARY_IMPL` Example) |Full features| Operator definition, operator adaptation, ATen IR registration and binding, meta function registration, and forward/backward bindings| Forward and backward bindings are implemented through `TORCH_LIBRARY_IMPL`.<br>For details, see the [Adaptation Development and Usage (Complete TORCH_LIBRARY_IMPL Example)](./cpp_extension_full/torch_lib_impl/README.md).|
|`cpp_extension_structured` (Structured) |Structured adaptation|  Operator definition, operator adaptation, and ATen IR registration and binding|                Adaptation is completed automatically through YAML configuration. For details, see the [Adaptation Development and Usage (Structured)](./cpp_extension_structured/README.md).               |
|`cpp_extension_asc` (AscendC) | Custom operator  | Operator definition, operator adaptation, and ATen IR registration and binding|                 Operators are written using Ascend C. For details, see the [Adaptation Development and Usage (AscendC-based)](./cpp_extension_asc/README.md).                |
|`cpp_extension_pybind` (pybind) | Flexible API | Operator definition, operator adaptation, and pybind binding|                       For details, see the [Adaptation Development and Usage (pybind-based)](./cpp_extension_pybind/README.md).                      |

> [!NOTE]
> 
> Prerequisites for using `cpp_extension_structured` (Structured): The ACLNN interface corresponding to `opapi` must be semantically consistent with the ATen IR, and the adaptation layer must contain no logic other than output tensor allocation.
