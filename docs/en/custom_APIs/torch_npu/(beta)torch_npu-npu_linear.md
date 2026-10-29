# (beta) torch_npu.npu_linear

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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

Multiplies matrix `a` by matrix `b` to produce matrix `a * b`.

## Prototype

```python
torch_npu.npu_linear(input, weight, bias=None) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Matrix tensor. The shape must have 2 dimensions. The data type can be `torch.float32`, `torch.float16`, `torch.int32`, or `torch.int8`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.

  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch.int32` and `torch.int8` data types are not supported.
  <!-- end id6 -->

- **`weight`** (`Tensor`): Required. Matrix tensor. The shape must have 2 dimensions. The data type can be `torch.float32`, `torch.float16`, `torch.int32`, or `torch.int8`. The data layout can be ND, `NHWC`, or `FRACTAL_NZ`.

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch.int32` and `torch.int8` data types are not supported.
  <!-- end id7 -->

- **`bias`** (`Tensor`): Optional. This parameter must be a 1D tensor. The data type can be `torch.float32`, `torch.float16`, or `torch.int32`. The data layout can be ND or `NHWC`. The default value is `None`.

  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch.int32` data type is not supported.
  <!-- end id8 -->

## Return Values

- **`out`** (`Tensor`): Output tensor. The data type is the same as that of the computation inputs `input` and `weight`. The data layout can be ND.

## Constraints

- This API supports only single-operator mode.

## Example

The single-operator call is as follows:

```python
>>> import torch
>>> import torch_npu
>>> x = torch.rand(2, 16).npu()
>>> w = torch.rand(4, 16).npu()
>>> b = torch.rand(4).npu()
>>> output = torch_npu.npu_linear(x, w, b)
>>> print(output)
tensor([[3.6335, 4.3713, 2.4440, 2.0081],
        [5.3273, 6.3089, 3.9601, 3.2410]], device='npu:0')
```
