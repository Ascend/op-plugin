# (beta) torch_npu.npu_linear

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Multiplies matrix `a` by matrix `b` to produce matrix `a * b`.

## Prototype

```python
torch_npu.npu_linear(input, weight, bias=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Matrix tensor. The shape must have 2 dimensions. The data type can be `float32`, `float16`, `int32`, or `int8`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.
- **`weight`** (`Tensor`): Required. Matrix tensor. The shape must have 2 dimensions. The data type can be `float32`, `float16`, `int32`, or `int8`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.
- **`bias`** (`Tensor`): Optional. This parameter must be a 1D tensor. The data type can be `float32`, `float16`, or `int32`. The data layout can be ND or `NHWC`. The default value is `None`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x=torch.rand(2,16).npu()
>>> w=torch.rand(4,16).npu()
>>> b=torch.rand(4).npu()
>>> output = torch_npu.npu_linear(x, w, b)
>>> print(output)
tensor([[3.6335, 4.3713, 2.4440, 2.0081],
        [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
```
