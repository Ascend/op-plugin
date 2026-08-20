# (beta) torch_npu.utils.npu_combine_tensors

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

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
