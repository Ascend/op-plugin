# (beta) torch_npu.contrib.function.matmul_transpose

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Replaces the native implementation with an NPU custom operator to improve performance.

## Prototype

```python
torch_npu.contrib.function.matmul_transpose(tensor1, tensor2)
```

## Parameters

- **`tensor1`** (`Tensor`): Required. First input tensor for matrix multiplication.
- **`tensor2`** (`Tensor`): Required. Second input tensor for matrix multiplication.

## Return Values

`Tensor`

Output tensor.

## Constraints

In dynamic shape scenarios, box transformation deltas are not supported due to operator limitations.

## Example

```python
>>> import torch
>>> import torch_npu
>>> from torch_npu.contrib.function import matmul_transpose
>>> tensor1 = torch.randn(68, 5, 75, 16).npu()
>>> tensor1.requires_grad = True
>>> tensor2 = torch.randn(68, 5, 75, 16).npu()
>>> tensor2.requires_grad = True
>>> output = matmul_transpose(tensor1, tensor2)
>>> output.sum().backward()
>>> print(output.shape)
torch.Size([68, 5, 75, 75])
```
