# (beta) torch_npu.npu.config.allow_internal_format

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Controls whether private formats are used. When set to True, private formats are allowed. When set to False, no tensor in a private format can be requested, preventing private formats from circulating in the adaptation layer.

This configuration directly affects the behavior of the following APIs:

- `torch_npu.npu_format_cast`/`torch_npu.npu_format_cast_`/`at_npu::native::npu_format_cast`: when set to `False`, conversion to a private format is not supported and the target private format is automatically degraded to the corresponding base format, with the data and shape unchanged. When set to `True`, the conversion to the private format is performed.
- Convolution operators (mainly affecting `torch.nn.Conv1d` and `torch.nn.Conv2d`; `torch.nn.Conv3d` is not affected): affects the implementation path of convolution operators. When set to `False` and JIT compilation is disabled, convolution operators are implemented through aclnn operators; otherwise, they are implemented through aclop operators. In most cases, setting this parameter to `False` is recommended to avoid triggering online compilation.

## Prototype

```python
torch_npu.npu.config.allow_internal_format = bool
```

## Parameters

Input `bool` value.

<!-- npu="A3,910b" id6 -->
- <term>Atlas A2 training products</term>/<term>Atlas A3 training products</term>: the default value is `False`.
<!-- end id6 -->
<!-- npu="910,310p" id7 -->
- <term>Atlas inference products</term>/<term>Atlas training products</term>: the default value is `True`.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950DT products</term>: the default value is `False`, and only `False` is supported.
<!-- end id8 -->

## Constraints

After the setting, only tensors created subsequently follow the configuration. Tensors that have already been created are not affected.

## Example

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.config.allow_internal_format = False
```
