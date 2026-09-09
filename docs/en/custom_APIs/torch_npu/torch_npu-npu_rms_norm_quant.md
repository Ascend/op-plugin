# torch_npu.npu_rms_norm_quant

> [!NOTICE]  
> This API is updated in this version. For details about the specific changes, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product| Supported|
| :---------------------- | :------: |
| <term>Ascend 950DT </term>                                                |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>                       |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                                        |    √    |
| <term>Atlas inference products</term>                                               |    √    |

## Function

- API description: Performs normalization and quantization operations commonly used in large language models. Compared with the LayerNorm operator, it removes the mean-subtraction step. The RmsNormQuant operator fuses the RmsNorm operator and the subsequent Quantize operator to reduce data transfer overhead.
- Formulas:
  
  $$
  quant\_in_i = \frac{x_i}{Rms(x)}g_i+b_i, where \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y = round((quant\_in * scale) + offset)
  $$
  The `round` operation in the preceding formula supports `"CAST_RINT"` mode.

## Prototype

```python
torch_npu.npu_rms_norm_quant(x, gamma, beta, scale, offset, epsilon=1e-06, dst_dtype='int8') -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor representing the source data tensor in the normalization process, $x$ in the formula. The data layout can be ND. The shape must have 1 to 8 dimensions. Non-contiguous tensors are supported. Empty tensors are not supported.
  - Atlas inference products and Atlas 200I/500 A2 inference products: The data type can be `float16`.
  - Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components, and Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`.
  - Ascend 950DT: The data type can be `float16`, `bfloat16`, or `float32`.

- **`gamma`** (`Tensor`): Required. Scaling tensor in the normalization process, $g$ in the formula. The shape must have 1 or 2 dimensions. If the shape is 1D, its size must be identical to the last dimension size of `x`. If the shape is 2D, the first dimension size must be 1, and the second dimension size must be identical to the last dimension size of `x`. The data type must be identical to that of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
  - Atlas inference products and Atlas 200I/500 A2 inference products: The data type can be `float16`.
  - Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components, and Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`.
  - Ascend 950DT: The data type can be `float16`, `bfloat16`, or `float32`.

- **`beta`** (`Tensor`): Required. Offset tensor in the normalization process, $b$ in the formula. The shape must have 1 or 2 dimensions, following the identical structural requirements as those of `gamma`. The data type must be identical to that of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
  - Atlas inference products and Atlas 200I/500 A2 inference products: The data type can be `float16`.
  - Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components, and Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`.
  - Ascend 950DT: The data type can be `float16`, `bfloat16`, or `float32`.

- **`scale`** (`Tensor`): Required. Scale tensor used during quantization to produce `y`, $scale$ in the formula. This parameter must be 1D with shape `(1,)`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. The value of this parameter must not be 0.
  - Atlas inference products and Atlas 200I/500 A2 inference products: The data type can be `float16`.
  - Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components, and Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`.
  - Ascend 950DT: The data type can be `float16`, `bfloat16`, or `float32`.

- **`offset`** (`Tensor`): Required. Offset tensor in the quantization process, $offset$ in the formula. Its shape must be identical to that of `scale`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported. 
  - Atlas inference products, Atlas 200I/500 A2 inference products, Atlas A2 training products/Atlas 800I A2 inference products/A200I A2 Box heterogeneous components, and Atlas A3 training products/Atlas A3 inference products: The data type can be `int8`.
  - Ascend 950DT: The data type can be `int8`, `int32`, `float16`, `bfloat16`, or `float32`.

- **`epsilon`** (`double`): Optional. Parameter used to prevent division-by-zero errors, $eps$ in the formula. The default value is `1e-6`. A small positive value is recommended.

- **`dst_dtype`** (`int`): Optional. Output quantization data type. The default value is `int8`. When set to `None`, this parameter is processed as `int8`. Valid values are `"int8"` or `"quint4x2"`.
  - Atlas inference products, Atlas 200I/500 A2 inference products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products: The value can be `int8` or `quint4x2`.
  - Ascend 950DT: The value can be `int8`, `quint4x2`, `float8_e4m3fn`, `float8_e5m2`, or `hifloat8`.

## Return Values
  
  `Tensor`
  
  Calculation result ($y$ in the formula), representing the final quantization output tensor. Its data type is specified by the `dst_dtype` parameter. When `dst_dtype` is `quint4x2`, the data type of `y` is `int32`, the last dimension of its shape is the last dimension of `x` divided by 8, and its other dimensions must match those of `x`, where each `int32` element contains eight `int4` results. In other scenarios, the shape of `y` must match that of the input `x`, and its data type is specified by `dst_dtype`.

## Constraints

- Atlas inference products: The tail-axis lengths of `x`, `y`, and `gamma` must be greater than or equal to 32 bytes.
- Atlas A2 training products/Atlas A2 inference products: When `dst_dtype` is `"quint4x2"`, the last dimension of `x`, `gamma`, and `beta` must be an even number, and the last dimension of `x` must be divisible by 8.
- Description of data types supported by different product models:
  
  - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products:

    | x | gamma | beta | scale | offset | epsilon | y |
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | float16   | float16       | float16       | float16       | int8           | double      |int8      |
    | bfloat16  | bfloat16      | bfloat16      | bfloat16      | int8           | double      |int8      |
    | float16   | float16       | float16       | float16       | int8           | double      |int32      |
    | bfloat16  | bfloat16      | bfloat16      | bfloat16      | int8           | double      |int32      |

  - Atlas inference series products and Atlas 200I/500 A2 inference series products:

    | x | gamma | beta | scale | offset | epsilon | y
    | --------- | ------------- | ------------- | ------------- | -------------- | --------- |--------- |
    | float16   | float16       | float16       | float16       | int8           | double      |int8      |
    | float16   | float16       | float16       | float16       | int8           | double      |int32      |

  - <term>Ascend 950DT</term>：

    | x | gamma | beta | scale | offset | epsilon | y |
    | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | float16 | float16 | float16 | float16 | int8 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | bfloat16 | bfloat16 | bfloat16 | bfloat16 | int8 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | float16 | float16 | float16 | float16 | float16 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | double | int8, int4, float8_e4m3fn, float8_e5m2, hifloat8 |
    | float32 | float32 | float32 | float32 | float32 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | float16 | float16 | float16 | float32 | int32 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | bfloat16 | bfloat16 | bfloat16 | float32 | int32 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | float16 | float16 | float16 | float32 | float32 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |
    | bfloat16 | bfloat16 | bfloat16 | float32 | float32 | double | int8, int32, float8_e4m3fn, float8_e5m2, hifloat8 |

## Examples

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
