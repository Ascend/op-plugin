# (beta) torch_npu.npu.clear_npu_overflow_flag

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- <term>Atlas inference products</term>: Supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Supported
<!-- end id3 -->

## Function

Clears the NPU overflow detection flag.

## Prototype

```python
torch_npu.npu.clear_npu_overflow_flag()
```

## Constraints

Effective only in saturation mode. In `INF_NAN` mode, the API only issues a warning and returns directly without clearing the values. You are advised to use [torch_npu.npu.utils.npu_check_overflow](./(beta)torch_npu-npu-utils-npu_check_overflow.md).

## Example

```python
import torch
import torch_npu

a = torch.Tensor([65535]).npu().half()
a = a + a
if torch_npu.npu.get_npu_overflow_flag():
    torch_npu.npu.clear_npu_overflow_flag()
```
