# torch_npu.matmul_checksum

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas training products</term>: Supported
<!-- end id3 -->

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

A `torch.bool` scalar on the NPU. A value of `True` indicates that an AI Core hardware fault has been detected.

## Constraints

- This API supports only scenarios where the device is NPU.
- When the CANN version is 9.2.0 or later and the input parameters `a`, `b`, and `c` are 2D Tensors, the `torch.bfloat16` and `torch.float32` data types are supported. In other scenarios, only `torch.bfloat16` is supported.

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
