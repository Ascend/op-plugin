# torch_npu.npu.aclnn.allow_hf32

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

Sets or queries whether convolution operators support HF32.

## Prototype

```python
torch_npu.npu.aclnn.allow_hf32:bool
```

## Parameters

**`bool`**: Specifies whether to enable or disable support for HF32.

## Return Values

A boolean indicator indicating whether `allow_hf32` is enabled. The default value is `True`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
>>> torch_npu.npu.aclnn.allow_hf32 = False
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
False
>>> torch_npu.npu.aclnn.allow_hf32 = True
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
```
