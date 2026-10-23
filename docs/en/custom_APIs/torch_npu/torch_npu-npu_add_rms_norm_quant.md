# torch_npu.npu_add_rms_norm_quant

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

- **API function**:
    RMSNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction. The torch_npu.npu_add_rms_norm_quant operator fuses the Add operator before RMSNorm and the Quantize operator after RMSNorm to reduce data movement.

- **Formulas**:
  - AddRMSNorm computation:

  $$
  x_i={x1}+{x2}
  $$

  $$
  y=\operatorname{RMSNorm}(x)=\frac{x}{\operatorname{RMS}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{RMS}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - Quantization computation:
      - When `div_mode` is True:

      $$
      y1=round((y/scales1)+zero\_points1)
      $$

      $$
      y2=round((y/scales2)+zero\_points2)
      $$
    - When `div_mode` is False:

      $$
      y1=round((y*scales1)+zero\_points1)
      $$

      $$
      y2=round((y*scales2)+zero\_points2)
      $$

## Prototype

```python
torch_npu.npu_add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, beta=None, scales2=None, zero_points2=None, *, axis=-1, epsilon=1e-06, div_mode=True, dst_type=None) -> (y1, y2, x)
```

## Parameters

- **`x1`** (`Tensor`): Required. Source data Tensor in the normalization process, $x1$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape can have 1 to 8 dimensions.

  <!-- npu="310p" id5 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float16`.
  <!-- end id5 -->
  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
  <!-- end id7 -->

- **`x2`** (`Tensor`): Required. Source data Tensor in the normalization process, $x2$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape can have 1 to 8 dimensions, and the shape and data type must be the same as those of `x1`.
- **`gamma`** (`Tensor`): Required. Weight Tensor in the normalization process, $gamma$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape can have 1 to 8 dimensions. The shape must be the same as the dimensions of `x1` to be normalized, and the data type must be the same as that of `x1`.
- **`scales1`** (`Tensor`): Required. Scales Tensor used to obtain `y1` in the quantization process, $scales1$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape must be the same as that of `gamma`. When the value of `div_mode` is True, the value of this parameter cannot be 0.

  <!-- npu="310p" id8 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float32`.
  <!-- end id8 -->
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
  <!-- end id10 -->

- **`zero_points1`** (`Tensor`): Optional. Offset Tensor used to obtain `y1` in the quantization process, $zero\_points1$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape must be the same as that of `gamma`.

  <!-- npu="310p" id11 -->
  - <term>Atlas inference products</term>: The data type can be `torch.int32`.
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.int32` or `torch.bfloat16`.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int32`, `torch.float32`, `torch.float16`, or `torch.bfloat16`.
  <!-- end id13 -->

- **`beta`** (`Tensor`): Optional. Bias term in the normalization process, $beta$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape and data type must be the same as those of `gamma`.
- **`scales2`** (`Tensor`): Optional. Scales Tensor used to obtain `y2` in the quantization process, $scales2$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The data type must be the same as that of `scales1`. The shape must be the same as that of `gamma`. When the value of `div_mode` is True, the value of this parameter cannot be 0. The default value is None.
- **`zero_points2`** (`Tensor`): Optional. Offset Tensor used to obtain `y2` in the quantization process, $zero\_points2$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The data type must be the same as that of `zero_points1`. The shape must be the same as that of `gamma`. The default value is None.
- **`axis`** (`int`): Optional. Element-wise axis to be quantized, and the other axes are broadcast. The specified axis index cannot exceed the number of dimensions of the input Tensor. Currently, only the default value -1 is supported, and any other value does not take effect.
- **`epsilon`** (`float`): Optional. Input $epsilon$ in the formulas, used to prevent division by zero. The data type is `double`. A small positive number is recommended. The default value is 1e-6.
- **`div_mode`** (`bool`): Optional. Parameter in the formulas that determines whether the quantization formulas use division. The parameter type is Python `bool`. The default value is True.
- **`dst_type`** (`int`): Optional. Specifies the data type of the quantization outputs `y1` and `y2`. The default value is `torch.int8`. When None is passed, it is processed as `torch.int8`.

  <!-- npu="A3,910b,310p" id14 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas inference products</term>: The supported values are None and `torch.int8`.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: The supported values are None, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn`.
  <!-- end id15 -->

## Return Values

- **`y1`** (`Tensor`): Quantization output Tensor, $y1$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape is the same as that of the input `x1`.

  <!-- npu="A3,910b,310p" id16 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas inference products</term>: The data type can be `torch.int8`.
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
  <!-- end id17 -->

- **`y2`** (`Tensor`): Quantization output Tensor, $y2$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The shape and data type are the same as those of the output `y1`.
- **`x`** (`Tensor`): Sum of `x1` and `x2`, $x$ in the formulas. The data layout can be ND. Non-contiguous Tensors are supported. The data type and shape are the same as those of the input `x1`.

## Constraints

- This API supports only single-operator mode.

<!-- npu="310p" id18 -->
- <term>Atlas inference products</term>: The number of elements in the last dimension of `x1` and `x2` cannot be less than 32. The number of elements in `gamma`, `beta`, `scales1`, `zero_points1`, `scales2`, and `zero_points2` cannot be less than 32.
<!-- end id18 -->
- Boundary value scenarios:

  <!-- npu="310p" id19 -->
  - <term>Atlas inference products</term>: The input cannot contain inf or nan.
  <!-- end id19 -->
  <!-- npu="A3,910b" id20 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When the input is inf, the output is inf. When the input is nan, the output is nan.
  <!-- end id20 -->

- Dimension boundaries:
  The size of each dimension in the shapes of the parameters `x1`, `x2`, `gamma`, `scales1`, `zero_points1`, `beta`, `scales2`, `zero_points2`, `y1`, `y2`, and `x` must not exceed 2147483647, the maximum value of `torch.int32`.

- Data type support for each product:

  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT products</term>:

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.int32` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.float16` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.float16` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.float16` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.int32` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.bfloat16` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.bfloat16` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e5m2`/`torch.float8_e4m3fn` | `torch.bfloat16` |
  <!-- end id21 -->
  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.int32` | `torch.float16` | `torch.int8` | `torch.int8` | `torch.float16` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8` | `torch.int8` | `torch.bfloat16` |
  <!-- end id22 -->
  <!-- npu="310p" id23 -->
  - <term>Atlas inference products</term>:

    | x1 | x2 | gamma | scales1 | scales2 | zero_points1 | zero_points2 | beta | y1 | y2 | x |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.int32` | `torch.float16` | `torch.int8` | `torch.int8` | `torch.float16` |
  <!-- end id23 -->

## Examples

```python
import math

import numpy as np
import torch

import torch_npu

def test_npu_add_rms_norm_quant():
    shape_list = [[[16, ], [16, ]],
                  [[2, 16], [16, ]],
                  [[2, 16], [2, 16]],
                  [[16, 32], [16, 32]],
                  [[16, 32], [32, ]],
                  [[2, 2, 2, 8, 16, 32], [2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 8, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 16, 32], [2, 2, 2, 2, 2, 16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 16, 32], [32, ]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [2, 2, 2, 2, 2, 8, 16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [16, 32]],
                  [[2, 2, 2, 2, 2, 8, 16, 32], [32, ]]]
    for item in shape_list:
        x_shape = item[0]
        quant_shape = item[1]
        x1 = torch.randn(x_shape, dtype=torch.float16)
        x2 = torch.randn(x_shape, dtype=torch.float16)
        gamma = torch.randn(quant_shape, dtype=torch.float16)
        beta = torch.randn(quant_shape, dtype=torch.float16)
        scales1 = torch.randn(quant_shape, dtype=torch.float32)
        zero_points1 = torch.randint(-10, 10, quant_shape, dtype=torch.int32)

        x1_npu = x1.npu()
        x2_npu = x2.npu()
        gamma_npu = gamma.npu()
        beta_npu = beta.npu()
        scales1_npu = scales1.npu()
        zero_points1_npu = zero_points1.npu()

        y1_v1, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu)
        y1_v2, _, x_out = torch_npu.npu_add_rms_norm_quant(x1_npu, x2_npu, gamma_npu, scales1_npu, zero_points1_npu, beta_npu) 

if __name__ == "__main__":
    test_npu_add_rms_norm_quant()
```
