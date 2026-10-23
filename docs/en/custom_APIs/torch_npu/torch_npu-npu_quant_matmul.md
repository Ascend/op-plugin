# torch_npu.npu_quant_matmul

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

- Performs quantized matrix multiplication, supporting at least 2D and at most 6D input.

- Formulas:

  In the formula, x1Scale, x2Scale, and yScale correspond to the parameters `pertoken_scale`, `scale`, and `y_scale`, respectively. x2Offset and yOffset are both provided by the parameter `offset`, and the specific meaning is determined by the quantization scenario. For details about quantization modes, see [Quantization Introduction](https://gitcode.com/cann/ops-math/blob/9.2.0/docs/en/context/quant_mode_introduction.md) in the *CANN Operator Library*.

  <!-- npu="A3,910b" id5 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    The K-C, K-T, T-C, T-T, G-B, and K-G quantization modes are supported. For the input and output data type combinations of different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-G quantization mode</strong></summary>

      - `x1` is `torch.int8`, `x2` is `torch.int32`, x1Scale is `torch.float32`, x2Scale is `torch.uint64`/`torch.int64`, and yOffset is `torch.float32`:

        $$
        out = ((x1 \mathbin{@} (x2 * \text{x2Scale})) + \text{yOffset}) * \text{x1Scale}
        $$

      - `x1` and `x2` are `torch_npu.int4`, x1Scale and x2Scale are `torch.float32`, x2Offset is `torch.float16`, and `out` is `torch.float16`/`torch.bfloat16` (pertoken-pergroup asymmetric quantization):

        $$
        out = \text{x1Scale} * \text{x2Scale} * (x1 \mathbin{@} x2 - x1 \mathbin{@} \text{x2Offset})
        $$

    </details>

    <details>

    <summary><strong>K-C and K-T quantization modes</strong></summary>

      - With x1Scale and without bias:

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale}
        $$

      - With x1Scale, where bias is `torch.int32` (no offset in this scenario):

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

      - With x1Scale, where bias is `torch.bfloat16`/`torch.float16`/`torch.float32` (no offset in this scenario):

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>T-C and T-T quantization modes</strong></summary>

      - Without x1Scale and bias:

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + \text{x2Offset}
        $$

      - When `bias` is `torch.int32`:

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} + \text{x2Offset}
        $$

      - When `bias` is `torch.bfloat16` or `torch.float32` (without `offset`):

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>G-B quantization mode</strong></summary>

      `x1` and `x2` are `torch.int8`, x1Scale and x2Scale are `torch.float32`, bias is `torch.float32`, and `out` is `torch.float16`/`torch.bfloat16` (pergroup-perblock quantization):

      $$
      out = (x1 \mathbin{@} x2) * \text{x1Scale} * \text{x2Scale} + bias
      $$

    </details>
  <!-- end id5 -->
  <!-- npu="310p" id6 -->
  - <term>Atlas inference products</term>:

    The K-C quantization mode is supported. For the input and output data type combinations of different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-C quantization mode</strong></summary>

      - With x1Scale and without bias:

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale}
        $$

      - With x1Scale, where bias is `torch.int32` (no offset in this scenario):

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

    </details>
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:

    The T-C, T-T, K-C, K-T, G-B, B-B, MX, T-CG, and K-G quantization modes are supported. For the input and output data type combinations of different quantization modes, see [Constraints](#constraints).

    <details>

    <summary><strong>K-G quantization mode</strong></summary>

      `x1` and `x2` are `torch_npu.int4`, x1Scale and x2Scale are `torch.float32`, x2Offset is `torch.float16`, and `out` is `torch.float16`/`torch.bfloat16` (pertoken-pergroup asymmetric quantization):

      $$
      out = \text{x1Scale} * \text{x2Scale} * (x1 \mathbin{@} x2 - x1 \mathbin{@} \text{x2Offset})
      $$

    </details>

    <details>

    <summary><strong>T-C and T-T quantization modes</strong></summary>

      - `x1` and `x2` are `torch.int8`, there is no x1Scale, x2Scale is `torch.int64`/`torch.uint64`, the optional parameter x2Offset is `torch.float32`, and the optional parameter bias is `torch.int32`:

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} + \text{x2Offset}
        $$

      - The parameters satisfy any of the following conditions:
        - `x1` and `x2` are `torch.int8`, there is no x1Scale, x2Scale is `torch.int64`/`torch.uint64`, and the optional parameter bias is `torch.int32`;
        - `x1` and `x2` are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`, there is no x1Scale, x2Scale is `torch.int64`/`torch.uint64`, and the optional parameter bias is `torch.float32`;
        - `x1` and `x2` are `torch_npu.int4`, there is no x1Scale, x2Scale is `torch.int64`/`torch.uint64`, and the optional parameter bias is `torch.int32`.

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale}
        $$

      - `x1` and `x2` are `torch.int8`, there is no x1Scale, x2Scale is `torch.bfloat16`/`torch.float32`, and the optional parameter bias is `torch.bfloat16`/`torch.float32`:

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} + bias
        $$

      - `x1` and `x2` are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`, x1Scale and x2Scale are `torch.float32`, and the optional parameter bias is `torch.float32`:

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>K-C and K-T quantization modes</strong></summary>

      - The parameters satisfy any of the following conditions:
        - `x1` and `x2` are `torch.int8`, x1Scale is `torch.float32`, x2Scale is `torch.bfloat16`/`torch.float32`, and the optional parameter bias is `torch.int32`;
        - `x1` and `x2` are `torch_npu.int4`, x1Scale is `torch.float32`, x2Scale is `torch.bfloat16`/`torch.float32`, and the optional parameter bias is `torch.int32`.

        $$
        out = (x1 \mathbin{@} x2 + bias) * \text{x2Scale} * \text{x1Scale}
        $$

      - The parameters satisfy any of the following conditions:
        - `x1` and `x2` are `torch.int8`, x1Scale is `torch.float32`, x2Scale is `torch.bfloat16`/`torch.float32`, and the optional parameter bias is `torch.bfloat16`/`torch.float32`;
        - `x1` and `x2` are `torch.int8`, x1Scale is `torch.float32`, x2Scale is `torch.float32`, and the optional parameter bias is `torch.float16`/`torch.float32`;
        - `x1` and `x2` are `torch_npu.int4`, x1Scale is `torch.float32`, x2Scale is `torch.bfloat16`/`torch.float32`, and the optional parameter bias is `torch.bfloat16`/`torch.float32`;
        - `x1` and `x2` are `torch_npu.int4`, x1Scale is `torch.float32`, x2Scale is `torch.float32`, and the optional parameter bias is `torch.float16`/`torch.float32`;
        - `x1` and `x2` are `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`, x1Scale and x2Scale are `torch.float32`, and the optional parameter bias is `torch.float32`.

        $$
        out = x1 \mathbin{@} x2 * \text{x2Scale} * \text{x1Scale} + bias
        $$

    </details>

    <details>

    <summary><strong>G-B, B-B, and MX quantization modes</strong></summary>

      $$
      out[m,n] = \sum_{j=0}^{kLoops-1} \left(\left(\sum_{k=0}^{gsK-1} \text{x1Slice} * \text{x2Slice}\right) * \left(\text{x1Scale}[m/gsM,j] * \text{x2Scale}[j,n/gsN]\right)\right) + bias[n]
      $$

      In the formula, gsM, gsN, and gsK represent groupSizeM, groupSizeN, and groupSizeK, respectively. x1Slice represents the vector of the mth row of x1 with a length of groupSizeK, and x2Slice represents the vector of the nth column of x2 with a length of groupSizeK. The K axis is sliced starting from j * groupSizeK, where j is in the range $[0, kLoops)$, $kLoops = \lceil K / groupSizeK \rceil$, and K is the length of the K axis. A final slice whose length is less than groupSizeK is supported. For the G-B, B-B, and MX quantization modes, `[groupSizeM, groupSizeN, groupSizeK]` supports only the values `[`

    </details>

    <details>

    <summary><strong>T-CG quantization mode</strong></summary>

      $$
      out = (x1 \mathbin{@} (x2 * \text{x2Scale})) * \text{yScale}
      $$

      In the formula, x1 is `torch.float8_e4m3fn`, x2 is `torch_npu.float4_e2m1fn_x2`, x2Scale is `torch.float16`/`torch.bfloat16`, yScale is `torch.int64`/`torch.uint64`, and the type of out is the same as the type of x2Scale. x2 is pergroup quantized, and the output is perchannel dequantized.

    </details>
  <!-- end id7 -->

## Prototype

```python
torch_npu.npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None, x1_dtype=None, x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None, y_scale=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Input tensor representing the left matrix in matrix multiplication. The data layout can be $ND$. The shape must have 2 to 6 dimensions.

    <!-- npu="310p" id8 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8`.
    <!-- end id8 -->
    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch.int32`. The `torch.int32` type indicates `torch_npu.int4` matrix multiplication, and each `torch.int32` value stores eight `torch_npu.int4` values.
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.int8` or `torch.int32`. The `torch.int32` type indicates `torch_npu.int4` matrix multiplication, and each `torch.int32` value stores eight `torch_npu.int4` values.
    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, or `torch.int32`.
      - `torch.int32` indicates `torch_npu.int4` matrix multiplication, where each `torch.int32` value stores eight `torch_npu.int4` values.
      - For `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`, the optional parameter `x1_dtype` must be set to the corresponding type; in this case, the dtype of x1 itself no longer takes effect, but the dtype of x1 must still be an 8-bit data type to ensure the shape is correct.
      - When the data type is `torch_npu.float4_e2m1fn_x2`, two float4 values are packed into one 8-bit value.
    <!-- end id11 -->

- **`x2`** (`Tensor`): Required. Input tensor representing the right matrix in matrix multiplication. The data layout can be $ND$. The shape must have 2 to 6 dimensions.

    <!-- npu="310p" id12 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8`.
    <!-- end id12 -->
    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch.int32` and must be identical to that of `x1` (`torch.int32` has the same meaning as in `x1`, indicating `torch_npu.int4` computation).
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.int8` or `torch.int32` and must be identical to that of `x1` (`torch.int32` has the same meaning as in `x1`, indicating `torch_npu.int4` computation).
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, `torch.float32`, or `torch.int32`.
        - For `torch_npu.hifloat8` and `torch_npu.float4_e2m1fn_x2`, the optional parameter `x2_dtype` must be set to the corresponding type; in this case, the dtype of x2 itself no longer takes effect, but the dtype of x2 must still be an 8-bit data type to ensure the shape is correct.
        - When the data type is `torch_npu.float4_e2m1fn_x2`, two float4 values are packed into one 8-bit value.
        - When the data type is `torch.float32`, `torch.float32` carries `torch_npu.float4_e2m1fn_x2` inputs. For details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md). This is supported only in FRACTAL_NZ scenarios.
        - In full quantization scenarios, when the data type is `torch.int8` (only T-C, T-T, K-C, or K-T quantization), `torch_npu.hifloat8` (T-C, T-T, K-C, K-T, G-B, or B-B quantization), `torch.float8_e4m3fn` (mx full quantization, T-C, T-T, K-C, K-T, G-B, or B-B quantization), or `torch_npu.float4_e2m1fn_x2` (mx full quantization), or in mx fake-quantization and T-CG fake-quantization scenarios with the `torch.float32` data type, the data format also supports FRACTAL_NZ, and the torch_npu.npu_format_cast API can be used to convert ND to the FRACTAL_NZ format.
    <!-- end id15 -->

- **`scale`** (`Tensor`): Required. Quantization scale factor. The data layout can be $ND$. To pass a `scale` of type `torch.int64`, call `torch_npu.npu_trans_quant_param` in advance to obtain an `torch.int64` `scale`.

    <!-- npu="310p" id16 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float32` or `torch.int64`. The shape must be 1D `(t,)`, where `t=1` or `n`, and `n` indicates the last dimension of `x2`.
    <!-- end id16 -->
    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32`, `torch.int64`, or `torch.bfloat16`. The shape must be 1D `(t,)`, where `t=1` or `n`, and `n` indicates the last dimension of `x2`.
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float32`, `torch.int64`, or `torch.bfloat16`. The shape must be 1D `(t,)`, where `t=1` or `n`, and `n` indicates the last dimension of `x2`.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.int64`, `torch.float16`, `torch.bfloat16`, or `torch.float8_e8m0fnu` (the optional parameter scale_dtype must be set to the corresponding type; in this case, the dtype of scale itself no longer takes effect, but the dtype of scale must still be an 8-bit data type to ensure the shape is correct). The shape can be 1D or multi-dimensional. In 1D scenarios, the shape must be `(t,)`, where t=1 or n, and n is the same as the n of x2. For the shape and dtype constraints in multi-dimensional scenarios, see [Constraints](#constraints).
    <!-- end id19 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`offset`** (`Tensor`): This parameter is required only when `scale` is 2D, in which case the data type must be `torch.float16` and the shape must be 2D and the same as that of `scale`. In other scenarios, this parameter is optional and is used to adjust the offset of the quantized value. The data type can be `torch.float32`. The data layout supports ND. The shape must be 1D `(t,)`, where `t=1` or `n`, and `n` is the same as the `n` of `x2`.

- **`pertoken_scale`** (`Tensor`): Optional. Used to scale the original values to match the quantized range. The data type can be `torch.float32`. The data layout supports ND. The shape must be 1D `(m,)`, where `m` is the same as the `m` of `x1` and indicates the second-to-last dimension of `x1`.

  <!-- npu="310p" id20 -->
  - <term>Atlas inference products</term> does not support `pertoken_scale` currently.
  <!-- end id20 -->
  <!-- npu="910b" id21 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float32`. The shape must be 1D `(m,)`, where `m` is the same as the `m` of `x1`.
  <!-- end id21 -->
  <!-- npu="A3" id22 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float32`. The shape must be 1D `(m,)`, where `m` is the same as the `m` of `x1`.
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32` or `torch.float8_e8m0fnu` (the optional parameter `pertoken_scale_dtype` must be set to the corresponding type; in this case, the dtype of `pertoken_scale` itself no longer takes effect, but the dtype of `pertoken_scale` itself must still be an 8-bit data type to ensure the shape is correct). The shape can be 1D or multi-dimensional. In 1D scenarios, the shape must be `(m,)` or `(1,)`. For the shape and dtype constraints in multi-dimensional scenarios, see [Constraints](#constraints).
  <!-- end id23 -->

- **`bias`** (`Tensor`): Optional. Bias term. The data layout supports ND. `n` is the same as the `n` of `x2`, and the `batch` value must be equal to the `batch` value derived after broadcasting `x1` and `x2`. When the output is 4D, 5D, or 6D, the shape of `bias` must be 1D `(n,)`. When the output is 3D, the shape of `bias` can be 1D `(n,)` or 3D `(batch, 1, n)`. When the output is 2D, `bias` of <term>Ascend 950PR&950DT products</term> supports 1D `(n,)` or 2D `(1, n)`, and other products support only 1D `(n,)`.
    <!-- npu="310p" id24 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int32`.
    <!-- end id24 -->
    <!-- npu="910b" id71 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int32`, `torch.bfloat16`, `torch.float16`, or `torch.float32`.
    <!-- end id71 -->
    <!-- npu="A3" id72 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.int32`, `torch.bfloat16`, `torch.float16`, or `torch.float32`.
    <!-- end id72 -->
    <!-- npu="950" id73 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int32`, `torch.bfloat16`, `torch.float16`, or `torch.float32`. The shape supports 1D `(n,)`, 2D `(1, n)`, or 3D `(batch, 1, n)`. Note that when the output is 2D, the shape of bias can be 1D or 2D.
    <!-- end id73 -->
- **`output_dtype`** (`int`): Optional. Data type of the output Tensor. The default value is `None`, which means the data type of the output Tensor is `torch.int8`.

    <!-- npu="310p" id25 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8` or `torch.float16`.
    <!-- end id25 -->
    <!-- npu="910b" id26 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id26 -->
    <!-- npu="A3" id27 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id27 -->
    <!-- npu="950" id28 -->
    - <term>Ascend 950PR&950DT products</term>: The input can be `torch.int8`, `torch.float16`, `torch.bfloat16`, `torch.float32`, or `torch.int32`.
    <!-- end id28 -->

- **`x1_dtype`** (`int`): Optional. Actual data type of `x1`. When a value is passed, the dtype of x1 itself is ignored and the data in x1 is treated as the type passed in for `x1_dtype` for computation. When no value is passed, the dtype of x1 is used directly for computation.

  <!-- npu="310p" id29 -->
  - <term>Atlas inference products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id29 -->
  <!-- npu="910b" id30 -->
  - <term>Atlas A2 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id30 -->
  <!-- npu="A3" id31 -->
  - <term>Atlas A3 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch_npu.float4_e2m1fn_x2` and `torch_npu.hifloat8` types are supported.
  <!-- end id32 -->

- **`x2_dtype`** (`int`): Optional. Actual data type of `x2`. When a value is passed, the dtype of x2 itself is ignored and the data in x2 is treated as the type passed in for `x2_dtype` for computation. When no value is passed, the dtype of x2 is used directly for computation.

  <!-- npu="310p" id33 -->
  - <term>Atlas inference products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id33 -->
  <!-- npu="910b" id34 -->
  - <term>Atlas A2 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id34 -->
  <!-- npu="A3" id35 -->
  - <term>Atlas A3 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch_npu.float4_e2m1fn_x2` and `torch_npu.hifloat8` types are supported.
  <!-- end id36 -->

- **`pertoken_scale_dtype`** (`int`): Optional. Actual data type of `pertoken_scale`. When a value is passed, the dtype of pertoken_scale itself is ignored and the data in pertoken_scale is treated as the type passed in for `pertoken_scale_dtype` for computation. When no value is passed, the dtype of pertoken_scale is used directly for computation.

  <!-- npu="310p" id37 -->
  - <term>Atlas inference products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id37 -->
  <!-- npu="910b" id38 -->
  - <term>Atlas A2 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id38 -->
  <!-- npu="A3" id39 -->
  - <term>Atlas A3 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch.float8_e8m0fnu` type is supported.
  <!-- end id40 -->

- **`scale_dtype`** (`int`): Optional. Actual data type of `scale`. When a value is passed, the dtype of scale itself is ignored and the data in scale is treated as the type passed in for `scale_dtype` for computation. When no value is passed, the dtype of scale is used directly for computation.

  <!-- npu="310p" id41 -->
  - <term>Atlas inference products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id41 -->
  <!-- npu="910b" id42 -->
  - <term>Atlas A2 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id42 -->
  <!-- npu="A3" id43 -->
  - <term>Atlas A3 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id43 -->
  <!-- npu="950" id44 -->
  - <term>Ascend 950PR&950DT products</term>: The `torch.float8_e8m0fnu` type is supported.
  <!-- end id44 -->

- **`group_sizes`** (`list[int]`): Optional. Group quantization granularity, used for the quantization group size in the $m$ (`x1` second-to-last dimension), $n$ (`x2` last dimension), and $k$ (`x1` last dimension/`x2` second-to-last dimension) directions. The format is [group_m, group_n, group_k]. The list must contain 3 elements, and each element must be 0 or a positive integer.

- **`y_scale`** (`Tensor`): Optional. Dequantizes the output result.

  <!-- npu="310p" id45 -->
  - <term>Atlas inference products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id45 -->
  <!-- npu="910b" id46 -->
  - <term>Atlas A2 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id46 -->
  <!-- npu="A3" id47 -->
  - <term>Atlas A3 products</term>: Reserved parameter, which cannot be set currently.
  <!-- end id47 -->
  <!-- npu="950" id48 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter is supported only when x1 is `torch.float8_e4m3fn` and x2 is `torch_npu.float4_e2m1fn_x2`. The shape is 2D `(1, n)`, where n is the same as the n of `x2`. The data type can be `torch.int64`. To pass `torch.int64` data, use torch_npu.npu_trans_quant_param to obtain a scale of the `torch.int64` data type. For parameter constraints, see [Constraints](#constraints).
  <!-- end id48 -->

## Return Values

`Tensor`

Represents the computation result of the quantized matmul. The shape supports 2 to 6 dimensions in the form `(batch, m, n)`, where batch may be absent. Broadcasting of the batch dimensions of `x1` and `x2` is supported. The output batch is the same as the batch after broadcasting, m is the same as the m of `x1`, and n is the same as the n of `x2`.

- If `output_dtype` is `torch.float16`, the output data type is `torch.float16`.
- If `output_dtype` is `torch.int8` or None, the output data type is `torch.int8`.

<!-- npu="950,A3,910b" id49 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
  - If `output_dtype` is `torch.bfloat16`, the output data type is `torch.bfloat16`.
  - If `output_dtype` is `torch.int32`, the output data type is `torch.int32`.
<!-- end id49 -->
<!-- npu="950" id50 -->
- <term>Ascend 950PR&950DT products</term>: If `output_dtype` is `torch.float32`, the output data type is `torch.float32`.
<!-- end id50 -->

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- **Common constraints**:
  - When the data format of `x2` needs to be FRACTAL_NZ, the ND format is usually converted to the FRACTAL_NZ format through `torch_npu.npu_format_cast`.
  - When the data type of the output `out` is `torch.int8` or `torch.float16` and there is no `pertoken_scale`, graph mode does not support passing `torch.float32` directly for `scale`.

<!-- npu="950" id51 -->
- Constraints for <term>Ascend 950PR&950DT products</term>:
  - **Empty Tensor description**: `x1`, `x2`, and `scale` usually cannot be empty Tensors. In special cases, if `x2` is in ND format, an empty Tensor is returned as the output for an empty Tensor with m or n = 0; if `x2` is in FRACTAL_NZ format, an empty Tensor is returned as the output for an empty Tensor with m = 0.
  - **Constraints related to x1 and x2**:
    - `x2` data format (full quantization): When the scenario is T-C, T-T, K-C, or K-T quantization (and the data type of `x2` is `torch.int8`, `torch_npu.hifloat8`, or `torch.float8_e4m3fn`), mx quantization (and the data type of `x2` is `torch.float8_e4m3fn` or `torch_npu.float4_e2m1fn_x2`), or G-B or B-B quantization (and the data type of `x2` is `torch.float8_e4m3fn` or `torch_npu.hifloat8`), converting `x2` to the FRACTAL_NZ format is supported. To convert `x2` to FRACTAL_NZ, none of the dimensions of the shapes of `x2` and `scale` may be 1 (in some special scenarios, 1 can work properly, but the FRACTAL_NZ feature is not guaranteed to be enabled).
    - `x2` data format (fake quantization): The scenarios in which `x2` supports the ND format include the eager mode of mx fake quantization and the eager mode, static graph mode, and dynamic graph mode of T-CG fake quantization. In the ND format, the data type of `x2` is `torch_npu.float4_e2m1fn_x2` (one `torch.uint8` or `torch.int8` value carries two `torch_npu.float4_e2m1fn_x2` values). The scenarios in which `x2` supports the FRACTAL_NZ format include the eager mode, static graph mode, and dynamic graph mode of mx fake quantization and T-CG fake quantization. In the FRACTAL_NZ format, `x2` supports `torch.float32` or `torch_npu.float4_e2m1fn_x2`: all FRACTAL_NZ scenarios support `torch.float32`, where one `torch.float32` value carries eight `torch_npu.float4_e2m1fn_x2` values; `torch_npu.float4_e2m1fn_x2` is supported only in mx fake-quantization scenarios.
  - **Constraints related to scale and pertoken_scale**:
    - In mx, G-B, and B-B quantization, the transposition of `scale` must be consistent with that of `x2`, and the transposition of `pertoken_scale` must be consistent with that of `x1`. In mx quantization, the transposition node must be written in the graph, the batch dimension of `scale` must be consistent with the batch dimension of `x2`, and the batch dimension of `pertoken_scale` must be consistent with the batch dimension of `x1`.
    - When the data types of the input `x1` and `x2` are `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8` and there is no `pertoken_scale`, graph mode does not support passing `torch.float32` directly for `scale`.
  - **Constraints related to batch consistency**:
    - B-B quantization scenarios do not support batch consistency. Even if the batch consistency switch is enabled, the output is not guaranteed to meet the batch consistency requirement.
    - In T-T and T-C quantization scenarios, if batch consistency must be satisfied, when inputs with different m values are compared and `pertoken_scale` is not empty, the input value of `pertoken_scale` must not change dynamically with the input value of `x1` and must remain unchanged.
<!-- end id51 -->
<!-- npu="A3,910b,310p" id52 -->
- <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term> constraints:
  - Empty Tensor description: The input `x1`, `x2`, and `scale` cannot be empty Tensors.
  - Constraints related to `x2`:
    - <term>Atlas inference products</term>: `x2` must be transposed first and then converted to FRACTAL_NZ.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: It is recommended to convert `x2` to FRACTAL_NZ without transposing it.
<!-- end id52 -->
<!-- npu="310p" id53 -->
- Data type constraints on parameters in <term>Atlas inference products</term> scenarios:

    **Table 1** Atlas inference accelerator cards

    | x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | `torch.float32`/None | `torch.int32`/None | None | `torch.int8` |
<!-- end id53 -->
<!-- npu="A3,910b" id54 -->
- Data type constraints on parameters in <term>Atlas A2 products</term> and <term>Atlas A3 products</term> scenarios:

    **Table 2** Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products

    | x1 | x2 | scale | offset | bias | pertoken_scale | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64`/`torch.float32` | `torch.float32`/None | `torch.int32`/None | None | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | `torch.int32`/`torch.bfloat16`/`torch.float32`/`torch.float16`/None | `torch.float32`/None | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32` | None | `torch.int32`/`torch.bfloat16`/`torch.float32`/`torch.float16`/None | `torch.float32` | `torch.float16` |
    | `torch.int32` | `torch.int32` | `torch.int64`/`torch.float32` | None | `torch.int32`/None | None | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | `torch.int32`/None | None | `torch.int32` |
<!-- end id54 -->
<!-- npu="950" id55 -->
- Data type constraints on parameters in each quantization scenario for <term>Ascend 950PR&950DT products</term>:

    The dtype, shape, and value of the `x1`, `x2`, `scale`, `pertoken_scale`, and `group_sizes` parameters affect each other in different quantization scenarios. Taking non-transposed `x1` and non-transposed `x2` as an example, the relationships are as follows.

    **Table 3** T-C and T-T quantization data type combinations<a id="table3"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int32` | `torch.int32` | `torch.int64`/`torch.uint64` | None | None | None/`torch.int32` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int64` | None | None | None/`torch.int32` | `torch.float16`/`torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.int64` | None | None/`torch.float32` | None/`torch.int32` | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | None | None | None/`torch.int32` | `torch.int32` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.int64` | None | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | None | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

    > **T-C and T-T quantization scenario description**:
    > - In T-T quantization scenarios, the shape of `pertoken_scale` is `(1,)` or None, and the shape of `scale` is `(1,)`.
    > - In T-C quantization scenarios, the shape of `pertoken_scale` is `(1,)` or None, and the shape of `scale` is `(n,)`, where n is the same as the n of `x2`.
    > - When the data types of `x1` and `x2` are `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`, static quantization and dynamic quantization are distinguished. In static quantization, the data type of `scale` is `torch.int64`; in dynamic quantization, the data type of `scale` is `torch.float32`. When the data types of `x1` and `x2` are `torch.int8` or `torch.int32`, dynamic T-C or dynamic T-T quantization is not supported.
    > - In static quantization scenarios, when `x1` and `x2` are `torch_npu.int4` or `torch.int32`, `x1` supports 2 to 6 dimensions, and `x2` supports only 2 dimensions.
    > - In static quantization scenarios, when the input types of both `x1` and `x2` are `torch_npu.hifloat8`, static graph mode and dynamic graph mode are supported when the data format of `x2` is ND; when the data format of `x2` is FRACTAL_NZ, only static graph mode is supported, and dynamic graph mode is not supported.

    **Table 4** K-C and K-T quantization data type combinations<a id="table4"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int32` | `torch.int32` | `torch.float32`/`torch.bfloat16` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int32` | `torch.int32` | `torch.float32` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.float16` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.float32`/`torch.bfloat16` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | None/`torch.int32`/`torch.float32`/`torch.float16` | `torch.float16` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

    > **K-C and K-T quantization scenario description**:
    > - In K-C quantization scenarios, the shape of `pertoken_scale` is `(m,)` and the shape of `scale` is `(n,)`, where m is the same as the m of `x1`, and n is the same as the n of `x2`.
    > - In K-T quantization scenarios, the shape of `pertoken_scale` is `(m,)` and the shape of `scale` is `(1,)`, where m is the same as the m of `x1`.
    > - When the data format of `x2` is FRACTAL_NZ, only non-transposed `x1` is supported.

    **Table 5** G-B and B-B quantization data type combinations<a id="table5"></a>

    | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | `torch.float32` | `torch.bfloat16` |

    **Table 6** Relationship between G-B and B-B quantization parameters, shape, and dtype<a id="table6"></a>

    | Quantization Mode | x1_dtype | x2_dtype | scale Data Type | pertoken_scale Data Type | y_scale Data Type | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes Value |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | B-B full quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, ceil(m/128), ceil(k/128)) | None | [128,128,128] |
    | B-B full quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, ceil(m/128), ceil(k/128)) | None | [128,128,128] |
    | G-B full quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |
    | G-B full quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |
    | G-B full quantization | `torch.int8` | `torch.int8` | `torch.float32` | `torch.float32` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/128), ceil(n/128)) | (batch, m, ceil(k/128)) | None | [1,128,128] |

    > **G-B and B-B quantization scenario description**:
    > - In G-B quantization scenarios, only `torch.int8` inputs support `bias`; other scenarios do not support `bias`.
    > - In B-B quantization scenarios, `torch.int8` inputs are not supported, and `bias` is not supported.
    > - Dimensions with a value of 0 in `group_sizes` are inferred automatically. The `group_sizes` values in the preceding table are those used when automatic inference is not enabled.
    > - When the data format of `x2` is FRACTAL_NZ, only non-transposed `x1` is supported.

    **Table 7** mx quantization data type combinations<a id="table7"></a>

    | Quantization Mode | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | mx full quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | mx full quantization | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | mx fake-quantization | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2`/`torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None/`torch.bfloat16`/`torch.float16` | `torch.bfloat16`/`torch.float16` |

    **Table 8** Relationship between mx quantization parameters, shape, and dtype<a id="table8"></a>

    | Quantization Mode | x1_dtype | x2_dtype | scale Data Type | pertoken_scale Data Type | y_scale Data Type | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes Value |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | mx full quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/64), n, 2) | (batch, m, ceil(k/64), 2) | None | [1,1,32] |
    | mx full quantization | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | (batch, m, k) | (batch, k, n) | (batch, ceil(k/64), n, 2) | (batch, m, ceil(k/64), 2) | None | [1,1,32] |
    | mx fake-quantization | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | (m, k) | (n, k/2) | (n, k/64, 2) | (m, k/64, 2) | None | [1,1,32] |
    | mx fake-quantization | `torch.float8_e4m3fn` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | (m, k) | (n, k/8) | (n, k/64, 2) | (m, k/64, 2) | None | [1,1,32] |

    > **mx quantization scenario description**:
    > - In mx full quantization scenarios, when the input types of both `x1` and `x2` are `torch_npu.float4_e2m1fn_x2`, the inner axis must be an even number, and k must be greater than 2.
    > - In mx full quantization scenarios, when the input types of both `x1` and `x2` are `torch_npu.float4_e2m1fn_x2` and the data format of `x2` is FRACTAL_NZ, only non-transposed `x1` is supported, and k or n must not be 1.
    > - In mx full quantization scenarios, when the input types of both `x1` and `x2` are `torch_npu.float4_e2m1fn_x2`, static graph mode and dynamic graph mode are supported when the data format of `x2` is ND; when the data format of `x2` is FRACTAL_NZ, only static graph mode is supported, and dynamic graph mode is not supported.
    > - In mx full quantization dynamic graph mode, when `x2` is `torch.float8_e4m3fn`/`torch.float8_e5m2` (ND format), `torch.float8_e4m3fn` (FRACTAL_NZ format), or `torch_npu.float4_e2m1fn_x2` (ND format), both of the following conditions must be satisfied: k is greater than 64, and neither the batch axis dimension of `x1` nor that of `x2` is 1.
    > - In mx full quantization scenarios, when `x2` is in FRACTAL_NZ format, the data types of `x1` and `x2` must both be `torch_npu.float4_e2m1fn_x2` or both be `torch.float8_e4m3fn`.
    > - In mx full quantization scenarios, only the last three axes of `scale` and `pertoken_scale` support non-contiguous Tensors.
    > - In mx fake-quantization scenarios, when the data type of `x1` is `torch.float8_e4m3fn` and the data type of `x2` is `torch_npu.float4_e2m1fn_x2`, only non-transposed `x1` and transposed `x2` are supported. If the data format of `x2` is ND, k must be a multiple of 8, and the values of k and n must not exceed 2<sup>31</sup>-1. Graph mode is not supported in this scenario. If the data format of `x2` is FRACTAL_NZ, k and n must be multiples of 8 and their values must not exceed 2<sup>31</sup>-1. Graph mode is supported in this scenario, but k must be greater than 64.
    > - In mx fake-quantization scenarios, when the data type of `x1` is `torch.float8_e4m3fn` and `x2_dtype` is `torch.float32`, only non-transposed `x1` and transposed `x2` are supported, and `x2` must be in FRACTAL_NZ format. The k values of `x1` and `x2` must be multiples of 8 and their values must not exceed 2<sup>31</sup>-1, and the n value of `x2` must be a multiple of 8 and must not exceed 2<sup>31</sup>-1. Graph mode is supported in this scenario, but k must be greater than 64.
    > - In mx fake-quantization scenarios, `bias` is optional. The data type can be `torch.bfloat16` or `torch.float16`, which must be the same as the output data type. The data layout supports ND, and the shape supports 2 dimensions in the form `(1, n)`.
    > - Dimensions with a value of 0 in `group_sizes` are inferred automatically. The `group_sizes` values in the preceding table are those used when automatic inference is not enabled.

    **Table 9** T-CG fake-quantization data type combinations<a id="table9"></a>

    | Quantization Mode | x1 | x2 | scale | pertoken_scale | offset | bias | output_dtype |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | T-CG fake-quantization | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2`/`torch.float32` | `torch.bfloat16`/`torch.float16` | None | None | None | `torch.bfloat16`/`torch.float16` |

    **Table 10** Relationship between T-CG fake-quantization parameters, shape, and dtype<a id="table10"></a>

    | x1_dtype | x2_dtype | scale Data Type | pertoken_scale Data Type | y_scale Data Type | x1 shape | x2 shape | scale shape | pertoken_scale shape | y_scale shape | group_sizes Value |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` | `torch.bfloat16`/`torch.float16` | None | `torch.int64` | (m, k) | (n, k/2) | (n, k/32) | None | (1, n) | [1,1,32] |
    | `torch.float8_e4m3fn` | `torch.float32` | `torch.bfloat16`/`torch.float16` | None | `torch.int64` | (m, k) | (k, n/8) | (k/32, n) | None | (1, n) | [1,1,32] |

    > **T-CG fake-quantization scenario description**:
    > - In the T-CG fake-quantization mode, the data type of `y_scale` can be `torch.int64`. The data layout supports ND, and the shape supports 2 dimensions in the form `(1, n)`.
    > - In T-CG fake-quantization scenarios, when the data type of `x1` is `torch.float8_e4m3fn` and `x2_dtype` is `torch_npu.float4_e2m1fn_x2`, `x2` must be in ND format, and only non-transposed `x1` and transposed `x2` are supported. The k values of `x1` and `x2` must be multiples of 32 and must not be 32, and their values must not exceed 2<sup>31</sup>-1. The n value of `x2` must not exceed 2<sup>31</sup>-1.
    > - In T-CG fake-quantization scenarios, when the data type of `x1` is `torch.float8_e4m3fn` and `x2_dtype` is `torch.float32`, `x2` must be in FRACTAL_NZ format, and only non-transposed `x1` and non-transposed `x2` are supported. The k values of `x1` and `x2` must be multiples of 32 and must not be 32, and their values must not exceed 2<sup>31</sup>-1. The n value of `x2` must be a multiple of 8 and must not exceed 2<sup>31</sup>-1.
    > - `bias` is not supported in T-CG fake-quantization scenarios.
    > - Dimensions with a value of 0 in `group_sizes` are inferred automatically. The `group_sizes` values in the preceding table are those used when automatic inference is not enabled.
<!-- end id55 -->
<!-- npu="A3,910b,310p" id74 -->
- **Additional constraints for `torch_npu.int4` computation**:

  Applies only to <term>Atlas inference products</term>, <term>Atlas A2 products</term>, and <term>Atlas A3 products</term>.

  - When the data types of both `x1` and `x2` are `torch.int32`, each `torch.int32` value stores eight `torch_npu.int4` values. For the input `torch.int32` shape, the last dimension must be reduced by a factor of 8 relative to the shape when the data is of the `torch_npu.int4` type. The last dimension of the shape of `torch_npu.int4` data must be a multiple of 8. For example, to perform an (m, k) x (k, n) matrix multiplication of the `torch_npu.int4` type, data of the `torch.int32` type with shapes (m, k//8) and (k, n//8) must be passed in, where both k and n must be multiples of 8. `x1` can accept only data with the shape (m, k//8) whose data layout is contiguous, and `x2` can accept either data with the shape (k, n//8) whose data layout is contiguous or data with the shape (k//8, n) transposed from contiguous data with the shape (n, k//8).
  - If this API is used in PyTorch graph mode and the environment variable `ENABLE_ACLNN=false`, the `x2` data with the shape (n, k//8) must be transposed before the API is called, and the transposition process must be written in the graph.

    > [!NOTE]  
    > A contiguous data layout means that all adjacent values in the array have contiguous memory addresses, including when the line changes. If the value returned by `Tensor.is_contiguous` is `True`, the Tensor data layout is contiguous.
<!-- end id74 -->

## Examples

- Single-operator call
  - Scenarios with `torch.int8` inputs:

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8)
    scale = torch.randn(N, dtype=torch.float32)
    offset = torch.randn(N, dtype=torch.float32)
    bias = torch.randint(-5, 5, (B, 1, N), dtype=torch.int32)
    # Method 1: You can directly call npu_quant_matmul
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset=offset.npu(), bias=bias.npu())

    # Method 2: You can first call npu_trans_quant_param to convert scale and offset from float32 to int64 when output dtype is not torch.bfloat16 and pertoken_scale is none
    scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu())
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_1,  bias=bias.npu())
    ```

  <!-- npu="950" id56 -->
  - `torch_npu.hifloat8` type + dual scale scenario. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8)
    scale_x2 = torch.randn(1, dtype=torch.float32)
    scale_x1 = torch.randn(1, dtype=torch.float32)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_x2.npu(), pertoken_scale=scale_x1.npu(), x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8, output_dtype=torch.float16)
    ```
  <!-- end id56 -->
  <!-- npu="950" id57 -->
  - `torch.float8_e4m3fn` type + pertoken_scale scenario. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    B = 31
    cpu_x1 = torch.randint(-5, 5, (1, M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    cpu_x2 = torch.randint(-5, 5, (B, K, N), dtype=torch.int8).to(torch.float8_e5m2)
    scale_x2 = torch.randn(N, dtype=torch.float32)
    scale_x1 = torch.randn(M, dtype=torch.float32)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu(), scale_x2.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.float16)
    ```
  <!-- end id57 -->
  <!-- npu="950" id58 -->
  - `torch.float8_e4m3fn` type + `torch_npu.float4_e2m1fn_x2` type + dual scale scenario. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    cpu_x1 = torch.randint(-5, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    # float4_e2m1fn_x2 data type can be represented by int8 or uint8
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    # float8_e8m0fnu data type can be represented by int8 or uint8
    scale_x2 = torch.randint(1, 5, (N, int(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(1, 5, (M, int(K/64), 2), dtype=torch.int8)
    bias = torch.randint(-5, 5, (1, N), dtype=torch.bfloat16)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), cpu_x2.npu().transpose(-1, -2), scale_x2.npu().transpose(-3,-2), bias=bias.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.bfloat16, x2_dtype=torch_npu.float4_e2m1fn_x2, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[0, 0, 32])
    ```
  <!-- end id58 -->
  <!-- npu="950" id59 -->
  - `torch.float8_e4m3fn` type + `torch_npu.float4_e2m1fn_x2` type + dual scale scenario + weightNZ scenario. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    M = 256
    K = 768
    N = 128
    cpu_x1 = torch.randint(-5, 5, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
    # float4_e2m1fn_x2 data type can be represented by int8 or uint8
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    npu_x2 = cpu_x2.npu()
    npu_x2 = torch_npu.npu_format_cast(npu_x2, 29, customize_dtype = torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)
    # float8_e8m0fnu data type can be represented by int8 or uint8
    scale_x2 = torch.randint(1, 5, (N, int(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(1, 5, (M, int(K/64), 2), dtype=torch.int8)
    bias = torch.randint(-5, 5, (1, N), dtype=torch.int16).to(torch.bfloat16)
    npu_out = torch_npu.npu_quant_matmul(cpu_x1.npu(), npu_x2.transpose(-1, -2), scale_x2.npu().transpose(-3,-2), bias=bias.npu(), pertoken_scale=scale_x1.npu(), output_dtype=torch.bfloat16, x2_dtype=torch_npu.float4_e2m1fn_x2, pertoken_scale_dtype=torch_npu.float8_e8m0fnu, scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[0, 0, 32])
    ```
  <!-- end id59 -->
  <!-- npu="950" id60 -->
  - `torch.float8_e4m3fn` + `torch_npu.float4_e2m1fn_x2` + FRACTAL_NZ scenario. The following is a code example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import torch_npu
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    from ml_dtypes import float8_e4m3fn

    m = 128
    k = 64
    n = 64
    group_sizes = 32
    E2M1_MIN, E2M1_MAX = -6, 6

    cpu_x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).to(torch.float8_e4m3fn)
    cpu_x2 = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(k * n).reshape((k, n))).astype(float4_e2m1fn)
    cpu_x2 = torch.from_numpy(cpu_x2.astype(np.float32))

    x2_scale = torch.randint(1, 5, (k//group_sizes, n), dtype=torch.bfloat16).npu()
    y_scale = torch.randint(-5, 5, (1, n), dtype=torch.float32).npu()
    x2_dtype = None
    bias = None
    pertoken_scale=None
    pertoken_scale_dtype = None
    scale_dtype = None

    npu_x2 = cpu_x2.npu()
    npu_x2 = torch_npu.npu_format_cast(npu_x2, 29, customize_dtype=torch.float8_e4m3fn)
    npu_x2 = torch_npu.npu_convert_weight_to_int4pack(npu_x2)
    # In per-group scenarios, convert `y_scale` from float32 to int64
    if (pertoken_scale == None) :
        y_scale = torch_npu.npu_trans_quant_param(y_scale)

    npu_out = torch_npu.npu_quant_matmul(
        cpu_x1.npu(),
        npu_x2,
        x2_scale,
        bias=bias,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=pertoken_scale_dtype,
        scale_dtype=scale_dtype,
        output_dtype=torch.bfloat16,
        x2_dtype=x2_dtype,
        group_sizes=[1, 1, group_sizes],
        y_scale=y_scale)
    ```
  <!-- end id60 -->
  <!-- npu="950" id61 -->
  - mx full quantization: `torch_npu.float4_e2m1fn_x2` + `torch_npu.float4_e2m1fn_x2` weightNZ type. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 64
    N = 7168
    K = 2112
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)
    x1_npu = cpu_x1.npu()
    x2_npu = cpu_x2.npu()
    x2_npu = torch_npu.npu_format_cast(x2_npu, 29)
    scale_x2_npu = scale_x2.npu()
    scale_x1_npu = scale_x1.npu()
    # Call the npu_quant_matmul function and specify x1_dtype and x2_dtype as torch_npu.float4_e2m1fn_x2
    npu_out = torch_npu.npu_quant_matmul(
    x1_npu,
    x2_npu.transpose(-1,-2),
    scale_x2_npu.transpose(0, 1),
    pertoken_scale=scale_x1_npu,
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    output_dtype=torch.float16,
    group_sizes=[1,1,32],
    x1_dtype=torch_npu.float4_e2m1fn_x2,
    x2_dtype=torch_npu.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu
    )
    print('npu_out is ', npu_out.cpu())
    ```
  <!-- end id61 -->
  <!-- npu="950" id62 -->
  - mx full quantization: `torch_npu.float4_e2m1fn_x2` + `torch_npu.float4_e2m1fn_x2` weightND type. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import math
    import torch
    import torch_npu
    import logging
    import os
    M = 256
    K = 768
    N = 16
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8)
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8)
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)

    x1_npu = cpu_x1.npu()
    x2_npu = cpu_x2.npu().transpose(-1,-2)
    scale_x2_npu = scale_x2.npu().transpose(0, 1)
    scale_x1_npu = scale_x1.npu()

    # Call the npu_quant_matmul function and specify x1_dtype and x2_dtype as torch_npu.float4_e2m1fn_x2
    npu_out = torch_npu.npu_quant_matmul(
        x1_npu,
        x2_npu,
        scale_x2_npu,
        pertoken_scale=scale_x1_npu,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        output_dtype=torch.float16,
        group_sizes=[1,1,32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu
    )
    ```
  <!-- end id62 -->

- Graph mode (ND layout)
  - Scenarios with `torch.float16` outputs:

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    # ENABLE_ACLNN specifies whether to use ACLNN. Valid values: true (uses ACLNN execution) or false (uses online compilation).
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M = 1
    K = 512
    N = 128
    B = 15
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias):
            return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, output_dtype=torch.float16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    cpu_x1 = torch.randint(-1, 1, (B, M, K), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (B, K, N), dtype=torch.int8)
    scale = torch.randn(1, dtype=torch.float32)
    # If pertoken_scale is not specified and the output data type is float16, call npu_trans_quant_param first to convert scale (offset) from float to int64.
    scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), None)
    bias = torch.randint(-1,1, (B, 1, N), dtype=torch.int32)
    # dynamic=True: dynamic graph mode; dynamic=False: static graph mode
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale_1, None, bias.npu())
    ```

  <!-- npu="A3,910b" id75 -->
  - Code sample for scenarios with `torch.bfloat16` outputs (supported on the following products):

    <!-- npu="910b" id63 -->
    - <term>Atlas A2 products</term>
    <!-- end id63 -->
    <!-- npu="A3" id64 -->
    - <term>Atlas A3 products</term>
    <!-- end id64 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2.t(), scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    m = 15
    k = 11264
    n = 6912
    bias_flag = True
    cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
    scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

    bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    if bias_flag:
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, bias.npu(), pertoken_scale.npu())
    else:
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, None, pertoken_scale.npu())
    ```
  <!-- end id75 -->

- Graph mode call (FRACTAL_NZ)

  <!-- npu="950" id65 -->
  - mx full quantization: `torch_npu.float4_e2m1fn_x2` + `torch_npu.float4_e2m1fn_x2` weightNZ type. The following is an example. Supported only by <term>Ascend 950PR&950DT products</term>

    ```python
    import torch
    import math
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    # Whether "ENABLE_ACLNN" enables the aclnn path. true: the callback goes through aclnn; false: online compilation
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2.transpose(-1,-2), scale.transpose(0,1), offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16,group_sizes=[1,1,32],
        x1_dtype=torch_npu.float4_e2m1fn_x2,
        x2_dtype=torch_npu.float4_e2m1fn_x2,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        scale_dtype=torch_npu.float8_e8m0fnu
        )
    cpu_model = MyModel()
    model = cpu_model.npu()
    M = 64
    N = 7168
    K = 2112
    cpu_x1 = torch.randint(-5, 5, (M, int(K/2)), dtype=torch.int8).npu()
    cpu_x2 = torch.randint(-5, 5, (N, int(K/2)), dtype=torch.int8).npu()
    scale_x2 = torch.randint(-5, 5, (N, math.ceil(K/64), 2), dtype=torch.int8)
    scale_x1 = torch.randint(-5, 5, (M, math.ceil(K/64), 2), dtype=torch.int8)
    cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2, 29)
    #cpu_x2_t_29 = cpu_x2
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    npu_out = model(cpu_x1, cpu_x2_t_29, scale_x2.npu(), None, None, scale_x1.npu())
    ```
  <!-- end id65 -->
  <!-- npu="950,310p" id76 -->
  - Graph mode call (high-performance data layout)

    <!-- npu="310p" id66 -->
    - <term>Atlas inference products</term>
    <!-- end id66 -->
    <!-- npu="950" id67 -->
    - <term>Ascend 950PR&950DT products</term>
    <!-- end id67 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    M = 1
    K = 512
    N = 128
    B = 15
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias):
            return torch_npu.npu_quant_matmul(x1, x2.transpose(2,1), scale, offset=offset, bias=bias)
    cpu_model = MyModel()
    model = cpu_model.npu()
    cpu_x1 = torch.randint(-1, 1, (B, M, K), dtype=torch.int8).npu()
    cpu_x2 = torch.randint(-1, 1, (B, K, N), dtype=torch.int8).npu()
    # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,n,k) layout
    cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2.transpose(2,1).contiguous(), 29)
    scale = torch.randn(1, dtype=torch.float32).npu()
    offset = torch.randn(1, dtype=torch.float32).npu()
    bias = torch.randint(-1,1, (N,), dtype=torch.int32).npu()
    # Process scale from float32 to int64 offline to improve performance
    scale_1 = torch_npu.npu_trans_quant_param(scale, offset)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=False)
    npu_out = model(cpu_x1, cpu_x2_t_29, scale_1, offset, bias)
    ```
  <!-- end id76 -->

  <!-- npu="950,A3,910b" id77 -->
  - Atlas A2 training products/Atlas A2 inference products

    <!-- npu="910b" id68 -->
    - <term>Atlas A2 products</term>
    <!-- end id68 -->
    <!-- npu="A3" id69 -->
    - <term>Atlas A3 products</term>
    <!-- end id69 -->
    <!-- npu="950" id70 -->
    - <term>Ascend 950PR&950DT products</term>
    <!-- end id70 -->

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
            return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
    cpu_model = MyModel()
    model = cpu_model.npu()
    m = 15
    k = 11264
    n = 6912
    bias_flag = True
    cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
    # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,k,n) layout
    x2_notranspose_29 = torch_npu.npu_format_cast(cpu_x2.npu().transpose(1,0).contiguous(), 29)
    scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)

    bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    if bias_flag:
        npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, bias.npu(), pertoken_scale.npu())
    else:
        npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, None, pertoken_scale.npu())
    ```
  <!-- end id77 -->
