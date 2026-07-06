# torch_npu.npu_quant_matmul_gelu

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E). 

## Supported Products

| Product                                           | Supported|
|-----------------------------------------------|:----:|
| <term>Atlas A3 training products/Atlas A3 inference products</term> |  √   |
| <term>Atlas A2 training products/Atlas A2 inference products</term> |  √   |

## Function

- Description: Performs fused computation of quantized matrix multiplication and the GELU activation function. It supports A8W8 and A4W4 quantization. This API integrates quantized matrix multiplication and GELU activation to reduce memory access and improve performance.

- Formulas:
    - Quantized matrix multiplication:
        - When `bias` is `int32`:
        $$
        qbmmout = (x1 \mathbin{@} x2 + \text{bias}) * x2Scale * x1Scale
        $$
        - When `bias` is `bfloat16`, `float16`, or `float32`:
        $$
        qbmmout = x1 \mathbin{@} x2 * x2Scale * x1Scale + \text{bias}
        $$
        - When `bias` is `None`:
        $$
        qbmmout = x1@x2 * x2Scale * x1Scale
        $$

    - GELU activation function, where the GELU type is specified by the input parameter `approximate`:
        - When `approximate` is `gelu_tanh`:
        $$
        out = gelu\_tanh(qbmmout)
        $$
        - When `approximate` is `"gelu_erf"` (default):
        $$
        out = gelu\_erf(qbmmout)
        $$

## Prototype

```python
torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, *, bias=None, approximate="gelu_erf") -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Input tensor representing the left matrix (activation values) in matrix multiplication. The data layout can be ND. The shape must have 2 to 6 dimensions. The data type can be `int8` (A8W8 quantization), `int32` (A4W4 quantization, where each `int32` element stores eight `int4` values), or `int4` (A4W4 quantization using `int4` representation).

- **`x2`** (`Tensor`): Required. Input tensor representing the right matrix (weights) in matrix multiplication. Its data type must be identical to that of `x1`. The data layout can be ND or NZ (Ascend-optimized layout). The shape must have 2 to 6 dimensions. The data type can be `int8` (A8W8 quantization), `int32` (A4W4 quantization, where each `int32` element stores eight `int4` values), or `int4` (A4W4 quantization using `int4` representation).<br>
In A8W8 quantization scenarios, the Ascend-optimized NZ layout is supported. You can use `torch_npu.npu_format_cast` to convert the data layout to NZ for improved performance.

- **`x1_scale`** (`Tensor`): Required. Quantization scaling factor for `x1`. The data layout can be ND. The data type can be `float32`. This parameter must be 1D with shape `(m,)`, where `m` must match the m dimension of `x1`. `pertoken` quantization is enabled, where each token has an independent scale value.

- **`x2_scale`** (`Tensor`): Required. Quantization scaling factor for `x2`. The data layout can be ND. The data type can be `float32` or `bfloat16`. This parameter must be 1D with shape `(n,)` or `(1,)`, where `n` must match the n dimension of `x2`. `perchannel` quantization is enabled, where each output channel has an independent scale value, or pertensor quantization is enabled when the shape is `(1,)`.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`bias`** (`Tensor`): Optional. Bias item. The data layout can be ND. The data type can be `int32`, `float32`, `bfloat16`, or `float16`.

    - In A4W4 quantization scenarios, this parameter must be 1D with shape `(n,)`, where `n` must match the n dimension of `x2`.
    - In A8W8 quantization scenarios, this parameter must be 1D with shape `(n,)` or 3D with shape `(batch, 1, n)`, where `n` must match the n dimension of `x2`. In addition, the `batch` value must be equal to the `batch` value derived after broadcasting `x1` and `x2`.

- `approximate` (`str`): Optional. Type of the GELU activation function. The default value is `"gelu_erf"`. Valid values are `"gelu_tanh"` (tanh approximation version of GELU) or `"gelu_erf"` (exact erf version of GELU).

## Return Values

`Tensor`

Output tensor representing the result of the fused quantized matrix multiplication and GELU activation.

- Rules for determining the output data type:
  - If the data type of `x2_scale` is `float32`, the output data type is `float16`.
  - If the data type of `x2_scale` is `bfloat16`, the output data type is `bfloat16`.
  - If the data type of `bias` is `bfloat16`, the output data type is forcibly set to `bfloat16` (the priority is higher than that of `x2_scale`).
- The output shape is `(batch, m, n)`, where `batch` is derived after broadcasting `x1` and `x2`.

## Constraints

- This API can be used in inference scenarios.
- The input parameters `x1`, `x2`, `x1_scale`, and `x2_scale` must not be empty tensors.
- The size of the last dimension of `x1` and `x2` must be less than or equal to `65535`.

- **Additional constraints for A4W4 quantization (`int4` or `int32` inputs)**:

    The A4W4 quantization scenarios support two input types:
    - **`int4`**: The `int4` data type is used directly.
    - **`int32`**: Each `int32` element stores eight `int4` values.
    
    When `int32` is used, the size of the last dimension of the input `int32` tensor must be reduced to one-eighth of that of the corresponding `int4` tensor. The size of the last dimension of the `int4` tensor must be a multiple of `8`.

    - The inner axis (`k` axis) of `x1` and `x2` must be an even number.
    - When the data type of `x2` is `int32`, its shape must be `(k, n // 8)`, where `n` must be a multiple of `8`.
    - When the data type of `x2` is `int4`, its shape must be `(k, n)`, where `n` must be a multiple of `8`.
    - A4W4 quantization supports only the ND layout and does not support the NZ layout.
    - Transposition information is automatically deduced by the operator based on the tensor stride and does not need to be specified manually.

- **Constraints for A8W8 quantization**

    - Both ND and NZ layouts are supported.
    - If the NZ layout is needed to improve performance, you can manually call `torch_npu.npu_format_cast` to complete the NZ layout conversion for the input `x2` (`weight`).
    - Transposition information is automatically deduced by the operator based on the tensor stride and does not need to be specified manually.

- The following tables describe the supported data type combinations among input parameters.

    **Table 1** Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products

    | x1    | x2    | x1_scale | x2_scale  | bias                                | Output Data Type   |
    |-------|-------|----------|-----------|-------------------------------------|-----------|
    | int8  | int8  | float32  | float32   | int32/float32/bfloat16/float16/None | float16   |
    | int8  | int8  | float32  | bfloat16  | int32/float32/bfloat16/float16/None | bfloat16  |
    | int32 | int32 | float32  | float32   | int32/None                          | float16   |
    | int32 | int32 | float32  | bfloat16  | int32/None                          | bfloat16  |
    | int4  | int4  | float32  | float32   | int32/None                          | float16   |
    | int4  | int4  | float32  | bfloat16  | int32/None                          | bfloat16  |

## Examples

- Single-operator call (A8W8 quantization, ND layout, `gelu_tanh` activation)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (A8W8 quantization, ND layout, `gelu_erf` activation, with bias)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>> bias = torch.randn(n, dtype=torch.float32).npu() * 0.1
    >>>
    >>> # Use the gelu_erf activation function and add bias.
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, bias=bias, approximate="gelu_erf"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (A8W8 quantization, NZ layout, `gelu_tanh` activation)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>>
    >>> # Convert x2 to the NZ layout to improve performance
    >>> x2_nz = torch_npu.npu_format_cast(x2.contiguous(), 29)  # 29 indicates ACL_FORMAT_FRACTAL_NZ
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # Automatically identify the NZ layout and call the corresponding API
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2_nz, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (A8W8 quantization, `bfloat16` output)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.bfloat16).abs().npu() * 0.01  # bfloat16 scale
    >>>
    >>> # The output data type is determined by the type of x2_scale. In this example, the output data type is bfloat16.
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.dtype)  # torch.bfloat16
    ```

- Single-operator call (A4W4 quantization)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> # Generate int4 data (stored in int32 format)
    >>> # Note: In actual use, float32 data must be quantized to int4 and packed into int32 using a quantization API.
    >>> x1 = torch.randint(-8, 8, (m, k // 8), dtype=torch.int32).npu()
    >>> x2 = torch.randint(-8, 8, (k, n // 8), dtype=torch.int32).npu()
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # A4W4 quantization supports only the ND layout and does not support the NZ layout.
    >>> # Transposition information is automatically deduced by the operator based on the tensor stride.
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, 
    ...     approximate="gelu_tanh"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (using the default setting `approximate="gelu_erf"`)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # If the approximate parameter is not specified, the default value "gelu_erf" is used.
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale)
    >>> print(output.dtype)  # torch.float16
    ```
