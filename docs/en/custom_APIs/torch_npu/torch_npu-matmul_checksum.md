# torch_npu.matmul_checksum

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Atlas A2 training products</term>  | √   |
| <term>Atlas training products</term>                                       |    √     |

## Function

Detects AI Core hardware faults based on native `torch.matmul` and `Tensor.matmul` APIs. This API internally verifies matrix computation results. It compares the verification error against a real-time computed verification threshold. If the verification error exceeds the threshold, this API raises an AI Core error.

## Prototype

```python
torch_npu.matmul_checksum(a, b, c) -> Tensor
```

## Parameters

- **`a`** (`Tensor`): Required. Input `input` for native matmul computation.
- **`b`** (`Tensor`): Required. Input `other` for native matmul computation.
- **`c`** (`Tensor`): Required. Output `out` of native matmul computation.

## Return Values

`Tensor`

Bool scalar on the NPU. A value of `True` indicates that an AI Core hardware fault has been detected.

## Constraints

This API supports only scenarios where the data type is `bfloat16` and the device is NPU.

## Example

   ```python
    >>> import torch
    >>> import torch_npu
    >>> matrix1 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
    >>> matrix2 = torch.randn(2000, 2000, device='npu', dtype=torch.bfloat16)
    >>> product = torch.matmul(matrix1, matrix2)
    >>> checksum = torch_npu.matmul_checksum(matrix1, matrix2, product)
    >>> print(checksum)
    tensor(False, device='npu:0')
   ```
