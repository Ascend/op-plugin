# torch_npu.npu_quant_matmul_gelu

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- Function: performs the fused computation of quantized matrix multiplication and the GELU activation function, supporting A8W8 and A4W4 quantization. This API fuses quantized matrix multiplication and the GELU activation, reducing memory access and improving performance.

- Formulas:
    - Quantized matrix multiplication computation:
        - When `bias` is `torch.int32`:
        $$
        qbmmout = (x1 \mathbin{@} x2 + \text{bias}) * x2Scale * x1Scale
        $$
        - When `bias` is `torch.bfloat16`/`torch.float16`/`torch.float32`:
        $$
        qbmmout = x1 \mathbin{@} x2 * x2Scale * x1Scale + \text{bias}
        $$
        - When `bias` is None:
        $$
        qbmmout = x1@x2 * x2Scale * x1Scale
        $$

    - GELU activation function, whose type is specified by the `approximate` input:
        - When `approximate` is `gelu_tanh`:
        $$
        out = gelu\_tanh(qbmmout)
        $$
        - When `approximate` is `gelu_erf` (default):
        $$
        out = gelu\_erf(qbmmout)
        $$

## Prototype

```python
torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, *, bias=None, approximate="gelu_erf") -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Input tensor representing the left matrix (activation values) in matrix multiplication. The data layout can be ND. The shape must have 2 to 6 dimensions. The data type can be `torch.int8` (A8W8 quantization), `torch.int32` (A4W4 quantization, where each `torch.int32` element stores eight `torch_npu.int4` values), or `torch_npu.int4` (A4W4 quantization using `torch_npu.int4` representation).

- **`x2`** (`Tensor`): Required. Input tensor representing the right matrix (weights) in matrix multiplication. Its data type must be identical to that of `x1`. The data layout can be ND or NZ (Ascend-optimized layout). The shape must have 2 to 6 dimensions. The data type can be `torch.int8` (A8W8 quantization), `torch.int32` (A4W4 quantization, where each `torch.int32` element stores eight `torch_npu.int4` values), or `torch_npu.int4` (A4W4 quantization using `torch_npu.int4` representation).<br>
In the A8W8 quantization scenario, the Ascend-friendly $NZ$ data layout is supported. It can be converted to the $NZ$ format using `torch_npu.npu_format_cast` to improve performance.

- **`x1_scale`** (`Tensor`): Required. Quantization scaling factor for `x1`. The data layout can be ND. The data type can be `torch.float32`. This parameter must be 1D with shape `(m,)`, where `m` must match the m dimension of `x1`. `pertoken` quantization is enabled, where each token has an independent scale value.

- **`x2_scale`** (`Tensor`): Required. Quantization scaling factor for `x2`. The data layout can be ND. The data type can be `torch.float32` or `torch.bfloat16`. This parameter must be 1D with shape `(n,)` or `(1,)`, where `n` must match the n dimension of `x2`. `perchannel` quantization is enabled, where each output channel has an independent scale value, or pertensor quantization is enabled when the shape is `(1,)`. The shape constraints are as follows:

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`bias`** (`Tensor`): Optional. Bias item. The data layout can be ND. The data type can be `torch.int32`, `torch.float32`, `torch.bfloat16`, or `torch.float16`.

    - In the A4W4 quantization scenario: only a 1D shape $(n,)$ is supported, where $n$ is the same as the $n$ of `x2`.
    - In the A8W8 quantization scenario: a 1D shape $(n,)$ or a 3D shape $(batch, 1, n)$ is supported, where $n$ is the same as the $n$ of `x2`, and the batch value must be equal to the batch value derived from broadcasting `x1` and `x2`.

- **`approximate`** (`str`): Optional. The default value is `"gelu_erf"`. Specifies the type of the GELU activation function. `"gelu_tanh"` (tanh approximation of GELU) and `"gelu_erf"` (erf exact version of GELU) are supported.

## Return Values

`Tensor`

Represents the computation result of quantized matrix multiplication fused with the GELU activation.

- Rules for determining the output data type:
  - If the data type of `x2_scale` is `torch.float32`, the output data type is `torch.float16`.
  - If the data type of `x2_scale` is `torch.bfloat16`, the output data type is `torch.bfloat16`.
  - If the data type of `bias` is `torch.bfloat16`, the output data type is forcibly `torch.bfloat16` (this takes precedence over `x2_scale`).
- The output shape is $(batch, m, n)$, where $batch$ is derived from the batch of `x1` and `x2` through broadcasting.

## Constraints

- This API can be used in inference scenarios.
- The passed `x1`, `x2`, `x1_scale`, and `x2_scale` must not be empty.
- The size of the last dimension of `x1` and `x2` must not exceed 65535.

- **Additional constraints for A4W4 quantization (`torch_npu.int4`/`torch.int32` type inputs)**:

    The A4W4 quantization scenario supports two input types:
    - **`torch_npu.int4` type**: the `torch_npu.int4` data type is used directly.
    - **`torch.int32` type**: each `torch.int32` value stores eight `torch_npu.int4` values.
    
    When the `torch.int32` type is used, the last dimension of the input `torch.int32` shape must be 1/8 of the last dimension of the shape when the data is of the original `torch_npu.int4` type. The last dimension of the `torch_npu.int4` data shape must be a multiple of 8.

    - The inner axis (k axis) of `x1` and `x2` must be an even number.
    - When `x2` is of the `torch.int32` type, the shape of `x2` is $(k, n//8)$, and $n$ must be a multiple of 8.
    - When `x2` is of the `torch_npu.int4` type, the shape of `x2` is $(k, n)$, and $n$ must be a multiple of 8.
    - A4W4 quantization supports only the $ND$ format, not the $NZ$ format.
    - The transpose information is automatically inferred by the operator based on the tensor stride and does not need to be specified manually.

- **Constraints for A8W8 quantization**:

    - The $ND$ and $NZ$ formats are supported.
    - If the $NZ$ format is required to improve performance, call `torch_npu.npu_format_cast` manually to convert the input `x2` (weight) to the $NZ$ format.
    - The transpose information is automatically inferred by the operator based on the tensor stride and does not need to be specified manually.

<!-- npu="A3,910b" id3 -->
- The supported data type combinations of the input parameters are as follows:

    **Table 1** <term>Atlas A2 products</term> and <term>Atlas A3 products</term>

    | x1 | x2 | x1_scale | x2_scale | bias | Output data type |
    |-------|-------|----------|-----------|-------------------------------------|-----------|
    | `torch.int8`  | `torch.int8`  | `torch.float32`  | `torch.float32`   | `torch.int32`/`torch.float32`/`torch.bfloat16`/`torch.float16`/None | `torch.float16`   |
    | `torch.int8`  | `torch.int8`  | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/`torch.float32`/`torch.bfloat16`/`torch.float16`/None | `torch.bfloat16`  |
    | `torch.int32` | `torch.int32` | `torch.float32`  | `torch.float32`   | `torch.int32`/None                          | `torch.float16`   |
    | `torch.int32` | `torch.int32` | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/None                          | `torch.bfloat16`  |
    | `torch_npu.int4`  | `torch_npu.int4`  | `torch.float32`  | `torch.float32`   | `torch.int32`/None                          | `torch.float16`   |
    | `torch_npu.int4`  | `torch_npu.int4`  | `torch.float32`  | `torch.bfloat16`  | `torch.int32`/None                          | `torch.bfloat16`  |
<!-- end id3 -->

## Example

- Single-operator call (A8W8 quantization, ND format, gelu_tanh activation)

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

- Single-operator call (A8W8 quantization, ND format, gelu_erf activation, with bias)

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
    >>> # Use the gelu_erf activation and add a bias
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, bias=bias, approximate="gelu_erf"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (A8W8 quantization, NZ format, gelu_tanh activation)

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>>
    >>> # Convert x2 to the NZ format to improve performance
    >>> x2_nz = torch_npu.npu_format_cast(x2.contiguous(), 29)  # 29 is ACL_FORMAT_FRACTAL_NZ
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # Automatically identify the NZ format and call the corresponding API
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2_nz, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (A8W8 quantization, `torch.bfloat16` output)

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
    >>> # The output data type is determined by the type of x2_scale. Here the output is bfloat16.
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
    >>> # Note: In actual use, float32 data must be quantized to int4 through a quantization API and packed into int32
    >>> x1 = torch.randint(-8, 8, (m, k // 8), dtype=torch.int32).npu()
    >>> x2 = torch.randint(-8, 8, (k, n // 8), dtype=torch.int32).npu()
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # A4W4 quantization supports only the ND format, not the NZ format
    >>> # The transpose information is automatically inferred by the operator based on the tensor stride
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, 
    ...     approximate="gelu_tanh"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- Single-operator call (using the default approximate="gelu_erf")

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
    >>> # Do not specify the approximate parameter and use the default value "gelu_erf"
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale)
    >>> print(output.dtype)  # torch.float16
    ```
