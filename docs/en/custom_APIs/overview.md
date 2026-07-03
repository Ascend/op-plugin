# Overview

Provides function descriptions, prototypes, parameter descriptions, and call examples for custom Ascend Extension for PyTorch APIs based on PyTorch versions such as 2.10.0, 2.9.0, 2.8.0, and 2.7.1.

The APIs provided by Ascend Extension for PyTorch comply with the public API conventions defined by the PyTorch community. For details, see [Public API definition and documentation](https://github.com/pytorch/pytorch/wiki/Public-API-definition-and-documentation). The APIs described in this document are the public APIs of Ascend Extension for PyTorch. Internal APIs may be modified or removed in future releases. Therefore, you are advised not to use them. If you must use them, submit an issue in the [Ascend community](https://gitcode.com/ascend/pytorch/issues) for assistance.

Ascend Extension for PyTorch integrates with PyTorch through monkey patching. Specifically, selected PyTorch APIs are dynamically replaced with Ascend Extension for PyTorch implementations, allowing users to continue using familiar PyTorch APIs on Ascend NPUs.

Ascend Extension for PyTorch is developed using both C++ and Python. Currently, only Python APIs are officially exposed. C++ APIs are intended for internal use and are not recommended for users.

Currently, some APIs are marked as beta APIs. Beta APIs are experimental and may exhibit unexpected behavior in certain scenarios. Exercise caution when using these APIs. We are committed to graduating beta APIs to stable APIs. However, before this process is complete, these APIs may still be changed as needed, including but not limited to parameter changes, renaming, and removal.

By default, all custom APIs support all PyTorch versions compatible with the corresponding Ascend Extension for PyTorch release. If an API does not support all compatible PyTorch versions, the restrictions are specified in the Constraints section of that API.

> [!NOTE]  
> Some features of Ascend Extension for PyTorch can be configured through environment variables. For details, see [Environment Variable Reference](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/environment_variable_reference/env_variable_list.md).
