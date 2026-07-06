# (beta) torch_npu.npu.config.allow_internal_format

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Controls whether to use private formats. When set to `True`, private formats are allowed. When set to `False`, allocation of tensors in any private format is not allowed, preventing private formats from being propagated in the adaptation layer.

## Prototype

```python
torch_npu.npu.config.allow_internal_format = bool
```

## Parameters

The input is a `bool` value.

- Atlas A2 training products/Atlas A3 training products The default value is `False`.
- Atlas inference products/Atlas training products The default value is `True`.

## Constraints

Once set, only subsequently created tensors follow this configuration. Already created tensors are not affected.

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.config.allow_internal_format = False
```
