# torch_npu.utils.get_cann_version

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas training products</term>  | √   |

## Function

Obtains the version number of CANN or related components in the current environment.

## Prototype

```python
torch_npu.utils.get_cann_version(module="CANN")
```

## Parameters

**`module`** (`str`): Optional. Specifies the component whose version number needs to be obtained. The default value is `"CANN"`. Valid values are `"CANN"`, `"RUNTIME"`, `"COMPILER"`, `"HCCL"`, `"TOOLKIT"`, `"OPP"`, `"OPP_KERNEL"`, or `"DRIVER"`.

Value description:

- `"CANN"`: Compute Architecture for Neural Networks (CANN) is a heterogeneous computing architecture launched by Ascend for AI scenarios.

- `"RUNTIME"`: Runtime component

- `"COMPILER"`: Compiler

- `"HCCL"`: Collective communication library

- `"TOOLKIT"`: Development toolkit

- `"OPP"`: Operator package

- `"OPP_KERNEL"`: Binary operator package

- `"DRIVER"`: Driver

## Return Values

`str`

Version number of the specified component.

An empty string is returned if the provided `module` is invalid.

## Constraints

1. The version number of `"DRIVER"` is obtained based on the information in `/etc/ascend_install.info` and `/usr/local/Ascend/driver/version.info`. You must ensure these two files are mapped inside the container when obtaining this version number within a container environment.
2. This feature is not supported and an empty string is returned when the CANN version is earlier than 8.1.RC1.

## Example

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.utils import get_cann_version
>>> version = get_cann_version(module="CANN")
>>> print(version)
'8.3.RC1'
```
