# (beta) torch_npu.npu_bmmV2

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.bmm` and `torch.view` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Multiplies matrix `a` by matrix `b` to produce matrix `a * b`. FakeTensor mode is supported.

## Prototype

```python
torch_npu.npu_bmmV2(self, mat2, output_sizes) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Matrix tensor. The shape must have 2 or more dimensions. The data type can be `float16`, `float32`, or `int32`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.
- **`mat2`** (`Tensor`): Matrix tensor. The shape must have 2 or more dimensions. The data type can be `float16`, `float32`, or `int32`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.
- **`output_sizes`** (`List[int]`): Output shape used for matmul backpropagation. The default value is `[]`.

## Examples

Example 1:

```python
>>> mat1 = torch.randn(10, 3, 4).npu()
>>> mat2 = torch.randn(10, 4, 5).npu()
>>> res = torch_npu.npu_bmmV2(mat1, mat2, [])
>>> print(res.shape)
torch.Size([10, 3, 5])
```

Example 2:

```python
// FakeTensor mode
>>> from torch._subclasses.fake_tensor import FakeTensorMode
>>> with FakeTensorMode():
...     mat1 = torch.randn(10, 3, 4).npu()
...     mat2 = torch.randn(10, 4, 5).npu()
...     result = torch_npu.npu_bmmV2(mat1, mat2, [])
...
>>> print(result)
FakeTensor(..., device='npu:0', size=(10, 3, 5))
```
