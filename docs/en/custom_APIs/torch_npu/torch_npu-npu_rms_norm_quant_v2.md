# torch_npu.npu_rms_norm_quant_v2

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: RMSNorm is a normalization operation commonly used in foundation models. Compared with LayerNorm, it removes the mean subtraction. RmsNormQuantV2 fuses RMSNorm and the Quantize operator after RMSNorm to reduce data movement operations. It also adds the `rstd` output based on RmsNormQuant.
- Formulas:
  - RMSNorm computation:

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - Quantization computation process:
    - When `div_mode` is True:

    $$
    y=round((quant\_in/scale)+offset)
    $$

    - When `div_mode` is False:

    $$
    y=round((quant\_in*scale)+offset)
    $$

## Prototype

```python
torch_npu.npu_rms_norm_quant_v2(x, gamma, scale, *, offset, beta, epsilon=1e-06, div_mode=True, dst_dtype=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input Tensor, which is the source data Tensor in the normalization process and corresponds to $x$ in the formulas. The data layout can be ND. The shape can have 1 to 8 dimensions. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. Non-contiguous Tensors are supported. Empty Tensors are not supported.

- **`gamma`** (`Tensor`): Required. Scaling Tensor in the normalization process, which corresponds to $gamma$ in the formulas. The data layout can be ND. The shape can be 1D or 2D. If the shape is 1D, it must be the same as the last dimension of `x`. If the shape is 2D, the first dimension must be 1, and the second dimension must be the same as the last dimension of `x`. The data type must be the same as that of `x`. Non-contiguous Tensors are supported. Empty Tensors are not supported.

- **`scale`** (`Tensor`): Required. Scales Tensor used to obtain `y` in the quantization process, which corresponds to $scale$ in the formulas. The data layout can be ND. The shape is 1 and the number of dimensions is 1. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. Non-contiguous Tensors are supported. Empty Tensors are not supported. The value cannot be 0.

- **`offset`** (`Tensor`): Optional. Offset Tensor in the quantization process, which corresponds to $offset$ in the formulas. The data layout can be ND. The shape must be the same as that of `scale`. The data type can be `torch.int8`, `torch.int32`, `torch.float16`, `torch.bfloat16`, or `torch.float32`. Non-contiguous Tensors are supported. Empty Tensors are not supported.

- **`beta`** (`Tensor`): Optional. Offset Tensor in the normalization process, which corresponds to $beta$ in the formulas. The shape, data type, and data layout must be the same as those of `gamma`. Non-contiguous Tensors are supported. Empty Tensors are not supported.

- **`epsilon`** (`float`): Optional. Corresponds to $epsilon$ in the formulas and is used to prevent division by zero. A small positive value is recommended. The default value is `1e-6`.

- **`div_mode`** (`bool`): Optional. Corresponds to $div\_mode$ in the formulas and determines whether division is used in the quantization formula. The default value is True.

- **`dst_dtype`** (`int`): Optional. Specifies the type of the quantized output `y`. The default value is None. The supported values are `torch.int8`, `torch.quint4x2`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, and `torch_npu.hifloat8`. When None is passed, it is treated as `torch.int8`.

## Return Values

- **`y`** (`Tensor`): Return result, which corresponds to $y$ in the formulas and is the final quantized output Tensor. The data type is specified by `dst_dtype`. When `dst_dtype` is `torch.quint4x2`, the data type of `y` is `torch.int32`, the last dimension of the shape is the last dimension of `x` divided by 8, and the other dimensions are the same as those of `x`. Each `torch.int32` element contains eight `torch_npu.int4` results. In other scenarios, the shape of `y` is the same as that of the input `x`.

- **`rstd`** (`Tensor`): Return result, which corresponds to the reciprocal of $Rms(x)$ in the formulas and is the reciprocal of the root mean square after normalization. The data type can be `torch.float32` only. The number of dimensions must be the same as that of `x`. The dimensions that do not require normalization have the same sizes as the corresponding dimensions of `x`, and the dimensions that require normalization have a size of 1. Example of the relationship among the shapes of `x`, `gamma`, and `rstd`: if `x` has shape `(2, 3, 4, 8)` and `gamma` has shape `(8,)`, `rstd` has shape `(2, 3, 4, 1)`.

## Constraints

- This API supports training and inference scenarios.
- This API supports only single-operator mode.
- When `dst_dtype` is `torch.quint4x2`, the last dimension of `x`, `gamma`, and `beta` must be even, and the last dimension of `x` must be divisible by 8.
- When the data type of `y` is `torch.int32`, the last dimension of `y` must be 1/8 of the last dimension of `x`.
- Supported data types:

    | x | gamma | scale | offset | beta | epsilon | y | rstd |
    | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.int8` | `torch.float16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.int8` | `torch.bfloat16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float32` | `torch.int32` | `torch.float16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.int32` | `torch.bfloat16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.float16` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.float16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.float32` | `torch.bfloat16` | `double` | `torch.int8`/`torch.int32`/`torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float32` |

- **In single-operator mode**, the operator determines whether to output a valid `rstd` by reading the `requires_grad` attribute of the input Tensor (`x`). `requires_grad` is a standard attribute of PyTorch Tensors, and its default value is False. When `requires_grad` is False, the operator does not write `rstd`, and the API returns an empty Tensor with shape `[0]`. In this case, `rstd` is an invalid placeholder output.

## Examples

```python
import torch
import torch_npu

x = torch.randn([8, 64], dtype=torch.float16, requires_grad=True).npu()
gamma = torch.randn([64], dtype=torch.float16).npu()
scale = torch.ones(1, dtype=torch.float16).npu()
offset = torch.zeros(1, dtype=torch.float16).npu()
beta = torch.randn([64], dtype=torch.float16).npu()
y_npu, rstd_npu = torch_npu.npu_rms_norm_quant_v2(x, gamma, scale, offset=offset, beta=beta, epsilon=1e-6, div_mode=True, dst_dtype=torch.int8)
y = y_npu.cpu()
rstd = rstd_npu.cpu()
print(f"y dtype = {y.dtype}, y shape = {y.shape}")
print(f"rstd dtype = {rstd.dtype}, rstd shape = {rstd.shape}")
```
