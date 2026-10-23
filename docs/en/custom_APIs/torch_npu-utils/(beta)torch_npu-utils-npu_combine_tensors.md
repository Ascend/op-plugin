# (beta) torch_npu.utils.npu_combine_tensors

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

Applies NPU-based tensor fusion to merge multiple NPU tensors into a single memory-contiguous tensor. Accessing the original tensors is redirected to the corresponding offset regions of the fused tensor.

## Prototype

```python
torch_npu.utils.npu_combine_tensors(list_of_tensor, require_copy_value=True) -> Tensor
```

## Parameters

- **`list_of_tensor`** (`List[Tensor]`): List of tensors to be fused.
- **`require_copy_value`** (`bool`): Specifies whether to copy the values of the original tensors to the corresponding offset addresses of the fused tensor. The default value is `True`.

## Return Values

`Tensor`

Fused new tensor.

## Constraints

All tensors in the `list_of_tensor` list must be memory-contiguous NPU tensors. The data types of these tensors must be identical.
