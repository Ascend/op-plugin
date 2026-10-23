# torch_npu.npu_rms_norm_quant

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
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->

## Function

- Function: RmsNormQuant is a normalization operation commonly used by large models. Compared with the LayerNorm operation, it removes the part that subtracts the mean. The RmsNormQuant operation fuses the RmsNorm operation and the Quantize operation after RmsNorm, reducing load and store operations.
- Formulas:
  
  $$
  quant\_in_i = \frac{x_i}{Rms(x)}g_i+b_i, where \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y = round((quant\_in * scale) + offset)
  $$
  The `round` operation in the preceding formulas supports the CAST_RINT mode.

## Prototype

```python
torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=1e-06, dst_dtype=torch.int8) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor representing the source data tensor in the normalization process, $x$ in the formula. The data layout can be ND. The shape must have 1 to 8 dimensions. Non-contiguous tensors are supported. Empty tensors are not supported.

  <!-- npu="310p,310b" id6 -->
  - <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be `torch.float16`.
  <!-- end id6 -->
  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id8 -->

- **`gamma`** (`Tensor`): Required. Scaling tensor in the normalization process, $g$ in the formula. The shape must have 1 or 2 dimensions. If the shape is 1D, its size must be identical to the last dimension size of `x`. If the shape is 2D, the first dimension size must be 1, and the second dimension size must be identical to the last dimension size of `x`. The data type must be identical to that of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.

  <!-- npu="310p,310b" id9 -->
  - <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be `torch.float16`.
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id11 -->

- **`beta`** (`Tensor`): Required. Offset tensor in the normalization process, $b$ in the formula. The shape must have 1 or 2 dimensions, following the identical structural requirements as those of `gamma`. The data type must be identical to that of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.

  <!-- npu="310p,310b" id12 -->
  - <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be `torch.float16`.
  <!-- end id12 -->
  <!-- npu="A3,910b" id13 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id14 -->

- **`scale`** (`Tensor`): Required. Scale tensor used during quantization to produce `y`, $scale$ in the formula. This parameter must be 1D with shape `(1,)`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The value of this parameter must not be 0.

  <!-- npu="310p,310b" id15 -->
  - <term>Atlas inference products</term> and <term>Atlas 200I/500 A2 inference products</term>: The data type can be `torch.float16`.
  <!-- end id15 -->
  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id17 -->

- **`offset`** (`Tensor`): Required. Offset tensor in the quantization process, $offset$ in the formula. Its shape must be identical to that of `scale`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. 

  <!-- npu="A3,910b,310p,310b" id18 -->
  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>: The data type can be `torch.int8`.
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.int32`, `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id19 -->

- **`epsilon`** (`float`): Optional. Parameter used to prevent division-by-zero errors, $eps$ in the formula. The default value is `1e-6`. A small positive value is recommended.
- **`dst_dtype`** (`int`): Optional. Output quantization data type. The default value is `torch.int8`. When set to `None`, this parameter is processed as `torch.int8`. Valid values are `"int8"` or `"quint4x2"`.

  <!-- npu="A3,910b,310p,310b" id20 -->
  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas A3 products</term>, and <term>Atlas A2 products</term>: The value can be `torch.int8` or `torch.quint4x2`.
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT products</term>: The value can be `torch.int8`, `torch.quint4x2`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`.
  <!-- end id21 -->

## Return Values
  
  `Tensor`

  The return result corresponds to $y$ in the formulas, that is, the final quantized output tensor. The data type is specified by `dst_dtype`. When `dst_dtype` is `torch.quint4x2`, the data type of `y` is `torch.int32`, the last dimension of the shape is the last dimension of `x` divided by 8, and the other dimensions are the same as those of `x`. Each `torch.int32` element contains eight `torch_npu.int4` results. In other scenarios, the shape of `y` is the same as that of the input `x`, and the data type is specified by `dst_dtype`.

## Constraints

<!-- npu="310p" id22 -->
- <term>Atlas inference products</term>: The tail-axis lengths of `x`, `y`, and `gamma` must be greater than or equal to 32 bytes.
<!-- end id22 -->
<!-- npu="910b" id23 -->
- <term>Atlas A2 products</term>: When `dst_dtype` is `torch.quint4x2`, the last dimension of `x`, `gamma`, and `beta` must be an even number, and the last dimension of `x` must be divisible by 8.
<!-- end id23 -->
- Data types supported by each product model:
  
  <!-- npu="A3,910b" id24 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:

    | x | gamma | beta | scale | offset | epsilon | y |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | `double`      |`torch.int8`      |
    | `torch.bfloat16`  | `torch.bfloat16`      | `torch.bfloat16`      | `torch.bfloat16`      | `torch.int8`           | `double`      |`torch.int8`      |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | `double`      |`torch.int32`      |
    | `torch.bfloat16`  | `torch.bfloat16`      | `torch.bfloat16`      | `torch.bfloat16`      | `torch.int8`           | `double`      |`torch.int32`      |
  <!-- end id24 -->
  <!-- npu="310p,310b" id25 -->
  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>:

    | x | gamma | beta | scale | offset | epsilon | y |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | `double`      |`torch.int8`      |
    | `torch.float16`   | `torch.float16`       | `torch.float16`       | `torch.float16`       | `torch.int8`           | `double`      |`torch.int32`      |
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT products</term>:

    | x | gamma | beta | scale | offset | epsilon | y |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.int8` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `double` | `torch.int8`, `torch_npu.int4`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.int32` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.int32` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `double` | `torch.int8`, `torch.int32`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8` |
  <!-- end id26 -->

## Example

```python
>>> import torch
>>> import torch_npu
>>> eps = 1e-6
>>> x = torch.randn(16, dtype=torch.float16).npu()
>>> gamma = torch.randn(16, dtype=torch.float16).npu()
>>> beta = torch.zeros(16, dtype=torch.float16).npu()
>>> scale = torch.ones(1, dtype=torch.float16).npu()
>>> offset = torch.zeros(1, dtype=torch.int8).npu()
>>> y = torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, eps)
>>> y.cpu().numpy()
    tensor([ 1, -1,  2,  0, -2,  1,  0,  1,  2,  0,  2,  0,  0,  0,  0,  0],
        device='npu:0', dtype=torch.int8)
```
