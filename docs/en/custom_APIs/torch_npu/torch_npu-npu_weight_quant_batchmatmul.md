# torch\_npu.npu\_weight\_quant\_batchmatmul<a name="ZH-CN_TOPIC_0000002231202136"></a>

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
<!-- npu="310p" id39 -->
- <term>Atlas inference products</term>: Supported
<!-- end id39 -->

## Function

- Performs matrix multiplication with quantization support for the `weight` input and the output. `pertensor`, `perchannel`, and `pergroup` quantization modes are supported.
- Formulas:

     $$
     y = x @ ANTIQUANT(weight) + bias
     $$
     In the formulas, $weight$ represents the input in the fake-quantization scenario, and its dequantization formula $\text{ANTIQUANT}(\text{weight})$ is:
     $$
     ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
     $$
     When `quant_scale` is provided, the output is quantized using the following formula:
     $$
     y = QUANT(x @ ANTIQUANT(weight) + bias) \\
     = (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset
     $$
     When `quant_scale` is `None`, the final computation result is directly output as:
     $$
     y = x @ ANTIQUANT(weight) + bias
     $$

Different products support different quantization modes, as follows:

<!-- npu="950" id4 -->
- <term>Ascend 950PR&950DT products</term>: pertensor, perchannel, pergroup, MX (MicroScaling)
<!-- end id4 -->
<!-- npu="A3" id5 -->
- <term>Atlas A3 products</term>: pertensor, perchannel, pergroup
<!-- end id5 -->
<!-- npu="910b" id6 -->
- <term>Atlas A2 products</term>: pertensor, perchannel, pergroup
<!-- end id6 -->
<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: pertensor, perchannel, pergroup
<!-- end id7 -->

## Prototype

```python
torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0, weight_dtype=None) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. The left matrix in matrix multiplication. $x$ in the formulas. The data layout can be ND. Transposed non-contiguous tensors are supported. This parameter must be 2D with shape $(M, K)$.
    <!-- npu="310p" id40 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id40 -->

    <!-- npu="910b" id8 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id8 -->
    <!-- npu="A3" id9 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id10 -->

- **`weight`** (`Tensor`): Required. The right matrix in matrix multiplication. $weight$ in the formulas. Transposed non-contiguous tensors are supported. This parameter must be 2D with shape $(K, N)$. The dimensions must match those of `x`. When the data layout is ND, using a transposed `weight` input is recommended in `perchannel` scenarios to improve performance.
    <!-- npu="310p" id44 -->
    - <term>Atlas inference products</term>: The data type can be `torch.int8`. The data layout can be ND or FRACTAL_NZ, where the FRACTAL_NZ format is valid only in graph mode and requires the torch_npu.npu_format_cast API to convert ND to FRACTAL_NZ. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435).
    <!-- end id44 -->

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch.int32` (where `torch.int32` carries `torch_npu.int4` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)). The data layout can be ND or FRACTAL_NZ.
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.int8` or `torch.int32` (where `torch.int32` carries `torch_npu.int4` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)). The data layout can be ND or FRACTAL_NZ.
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.int32` (where `torch.int32` carries `torch_npu.int4` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)), `torch.float32` (where `torch.float32` carries `torch_npu.float4_e2m1fn_x2` inputs; for details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)), `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. The `torch.float8_e4m3fn` and `torch_npu.hifloat8` data types are supported only in perchannel scenarios. Note that the FRACTAL_NZ format supports `torch.int8`, `torch.int32`, and `torch.float32`. When the data type is `torch_npu.int4` or `torch_npu.float4_e2m1fn_x2`, the size of the last axis must be aligned to 8.
      - For different fake-quantization algorithm modes, the FRACTAL_NZ `weight` data format is supported only in the following scenarios:
        - perchannel mode: The data type of `weight` is `torch_npu.int4`/`torch.int32`, `weight` is not transposed, and `x` is not transposed. The data type of `weight` is `torch.int8`, and `x` is not transposed.
        - pergroup mode: The data type of `weight` is `torch_npu.int4`/`torch.int32`/`torch_npu.float4_e2m1fn_x2`/`torch.float32`/`torch.int8`, `weight` is not transposed, and `x` is not transposed.
        - MX mode: The data type of `weight` is `torch_npu.float4_e2m1fn_x2`/`torch.float32`, `weight` is not transposed, and `x` is not transposed.
    <!-- end id13 -->

- **`antiquant_scale`** (`Tensor`): Required. Dequantization scaling factor used to dequantize the weight matrix, corresponding to $antiquantScale$ in the dequantization formula. The data layout supports ND. Non-contiguous tensors with transpose are supported. The shapes supported by `antiquant_scale` depend on the quantization mode:

    - pertensor mode: The input shape is `(1,)` or `(1, 1)`.
    - perchannel mode: The input shape is `(1, N)` or `(N,)`.
    - pergroup mode: The input shape is `(ceil(K, antiquant_group_size), N)`.
    - MX mode: The input shape is `(⌈K/antiquant_group_size⌉, N)`, where `antiquant_group_size` indicates the size of each group into which k is divided, and only 32 is supported.

    The dtype supported by `antiquant_scale` is as follows:

    <!-- npu="310p" id45 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, which must be the same as that of `x`.
    <!-- end id45 -->

    <!-- npu="910b" id14 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int64`.
        - If the input is `torch.float16` or `torch.bfloat16`, its data type must be the same as that of `x`.
        - If the input is `torch.int64`, the data type of `x` must be `torch.float16` and the input must not be transposed. In addition, the data type of `weight` must be `torch.int8`, the data layout must be ND, and the input must be transposed. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435). In this case, only the perchannel scenario is supported, the value range of M is [1, 96], and K and N must be aligned to 64.
    <!-- end id14 -->
    <!-- npu="A3" id15 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int64`.
        - If the input is `torch.float16` or `torch.bfloat16`, its data type must be the same as that of `x`.
        - If the input is `torch.int64`, the data type of `x` must be `torch.float16` and the input must not be transposed. In addition, the data type of `weight` must be `torch.int8`, the data layout must be ND, and the input must be transposed. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435). In this case, only the perchannel scenario is supported, the value range of M is [1, 96], and K and N must be aligned to 64.
    <!-- end id15 -->
    <!-- npu="950" id16 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16` or `torch.bfloat16`, and must be the same as that of `x`. In MX scenarios, the type of `antiquant_scale` must be `torch.uint8` to carry `torch.float8_e8m0fnu` data; in this case, the type of `x` is `torch.float16` or `torch.bfloat16`.
    <!-- end id16 -->

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset used for dequantizing the `weight` matrix. $antiquantOffset$ in the dequantization formula. The default value is `None`. The data layout can be ND. Transposed non-contiguous tensors are supported. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`.
    <!-- npu="310p" id46 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`, which must match that of `antiquant_scale`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
    <!-- end id46 -->

    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int32`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
        - If the input is `torch.float16` or `torch.bfloat16`, its data type must match that of `antiquant_scale`.
        - If the input is `torch.int32`, the data type of `antiquant_scale` must be `torch.int64`.
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int32`. In `pergroup` scenarios, the shape must be `(ceil_div(K, antiquant_group_size), N)`.
        - If the input is `torch.float16` or `torch.bfloat16`, its data type must match that of `antiquant_scale`.
        - If the input is `torch.int32`, the data type of `antiquant_scale` must be `torch.int64`.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16` or `torch.bfloat16`, and must be the same as that of `x`. When `weight` is of a floating-point type, `antiquant_offset` must be None.
    <!-- end id19 -->

- **`quant_scale`** (`Tensor`): Optional. Quantization scale factor used for quantizing the output matrix. The default value is `None`. This parameter is supported only when the data layout of `weight` is ND. The data type can be `torch.float32` or `torch.int64`. The data layout can be ND. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`. When the data type of `antiquant_scale` is `torch.int64`, this parameter must be omitted.
    <!-- npu="310p" id47 -->
    - <term>Atlas inference products</term>: This parameter is not supported currently.
    <!-- end id47 -->

    <!-- npu="950" id20 -->
    - <term>Ascend 950PR&950DT products</term>: This parameter is not supported currently.
    <!-- end id20 -->

- **`quant_offset`** (`Tensor`): Optional. Quantization offset used for quantizing the output matrix. $quantOffset$ in the quantization formula. The default value is `None`. This parameter is supported only when the data layout of `weight` is ND. The data type can be `torch.float32`. The data layout can be ND. This parameter can be 1D with shape `(N,)` or `(1,)`, or 2D with shape `(1, N)`. When the data type of `antiquant_scale` is `torch.int64`, this parameter must be omitted.
    <!-- npu="310p" id48 -->
    - <term>Atlas inference products</term>: This parameter is not supported currently.

    <!-- end id48 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT products</term>: This parameter is not supported currently.
    <!-- end id21 -->

- **`bias`** (`Tensor`): Optional. Bias term in the matrix multiplication, corresponding to $bias$ in the formula. The default value is `None`. The data layout can be ND. Non-contiguous tensors are not supported. This parameter can be 2D with shape `(1, N)` or 1D with shape `(N,)`.
    <!-- npu="310p" id49 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id49 -->

    <!-- npu="910b" id22 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.float32`. When the data type of `x` is `torch.bfloat16`, the data type of `bias` must be `torch.float32`. When the data type of `x` is `torch.float16`, the data type of `bias` must be `torch.float16`.
    <!-- end id22 -->
    <!-- npu="A3" id23 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16` or `torch.float32`. When the data type of `x` is `torch.bfloat16`, the data type of `bias` must be `torch.float32`. When the data type of `x` is `torch.float16`, the data type of `bias` must be `torch.float16`.
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`. When the data type of `x` is `torch.bfloat16`, `bias` can be `torch.float32` or `torch.bfloat16`. When the data type of `x` is `torch.float16`, `bias` must be `torch.float16`. When the data type of `x` is `torch.bfloat16` and the data type of `weight` is `torch.float8_e4m3fn`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, or `torch.float8_e5m2`, `bias` must be `torch.bfloat16`. When the data type of `x` is `torch.bfloat16` and the data type of `weight` is `torch.int8` in FRACTAL_NZ format, `bias` must be `torch.float32`.
    <!-- end id24 -->

- **`antiquant_group_size`** (`int`): Optional. Used to control the group size in pergroup scenarios. It does not take effect in other quantization scenarios. The default value is 0.
  - In pergroup scenarios, the value range is [32, K-1] and must be a multiple of 32.
  - In MX scenarios, `antiquant_group_size` must be 32.
- **`inner_precise`** (`int`): Optional. Computation mode. The default value is 0. `0` indicates the high-precision mode, and `1` indicates the high-performance mode, which may affect accuracy. In pergroup scenarios where `weight` is input in the `torch.int32` type and the FRACTAL_NZ format and M is no greater than 16, this parameter can be set to 1 to improve performance. The high-performance mode is not recommended in other scenarios.
- **`weight_dtype`** (`int`): Optional. Extended data type of `weight` in torch_npu.

  <!-- npu="310p" id25 -->
  - <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id25 -->
  <!-- npu="910b" id26 -->
  - <term>Atlas A2 products</term>: This parameter is not supported currently.
  <!-- end id26 -->
  <!-- npu="A3" id27 -->
  - <term>Atlas A3 products</term>: This parameter is not supported currently.
  <!-- end id27 -->
  <!-- npu="950" id28 -->
  - <term>Ascend 950PR&950DT products</term>: Only `torch_npu.hifloat8` is supported.
  <!-- end id28 -->

## Return Values<a name="en-us_topic_0000001771071862_section22231435517"></a>

`Tensor`

When `quant_scale` is provided, the data type of the output must be `torch.int8`. If `quant_scale` is not provided, the data type of the output is identical to that of the input `x`.

## Constraints<a name="en-us_topic_0000001771071862_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.

  <!-- npu="A3,910b,310p" id29 -->
  - <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Note that when the input `weight` is in FRACTAL_NZ format, single-operator calls are not supported currently, and only graph mode calls are supported.
  <!-- end id29 -->

- This API supports graph mode. When the data layout of the input `weight` is FRACTAL_NZ, single-operator calls are currently not supported. Only graph mode calls are supported.

  <!-- npu="A3,910b,310p" id30 -->
  - <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The value ranges of K and N are [1, 65535]. When `x` is not transposed, the value range of M is [1, 2^31-1]; when `x` is transposed, the value range of M is [1, 65535].
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT products</term>: The value ranges of K and N are [1, 2^31-1]. When `x` is not transposed, the value range of M is [1, 2^31-1]; when `x` is transposed, the value range of M is [1, 2^31-1].
  <!-- end id31 -->

- Empty tensor inputs are not supported.
- The input shapes of `antiquant_scale` and `antiquant_offset` must be identical.
- The input shapes of `quant_scale` and `quant_offset` must be identical, and `quant_offset` cannot exist independently of `quant_scale`.
- To pass a `quant_scale` with a data type of `torch.int64`, you must call the `torch_npu.npu_trans_quant_param` API in advance to convert the `quant_scale` and `quant_offset` from a data type of `torch.float32` into a `quant_scale` input with a data type of `torch.int64`. For details, see [Examples](#en-us_topic_0000001771071862_section14459801435).
- When the input `weight` has a data layout of FRACTAL_NZ and the data type is `torch.int32`, `weight` must be transposed in `perchannel` scenarios; whereas in `pergroup` scenarios, `x` must be transposed, `weight` must not be transposed, `antiquant_group_size` must be `64` or `128`, `K` must be aligned to `antiquant_group_size`, and `N` must be aligned to 64.

  <!-- npu="A3,910b,310p" id32 -->
  - <term>Atlas inference products</term>, <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In perchannel scenarios, `weight` must be transposed. In pergroup scenarios, `x` must be transposed, `weight` must not be transposed, `antiquant_group_size` must be 64 or 128, K must be aligned to `antiquant_group_size`, and N must be aligned to 64.
  <!-- end id32 -->
  <!-- npu="950" id33 -->
  - <term>Ascend 950PR&950DT products</term>: In pergroup scenarios, `x` and `weight` must not be transposed. The value range of `antiquant_group_size` is [32, K-1] and must be a multiple of 32.
  <!-- end id33 -->

- When the type of the input `weight` is `torch.int8`, `torch.float8_e4m3fn`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, or `torch.float32`, K and N do not need to be aligned to 32 bytes.
- When the input `weight` shape is `(1, 8)` and the data type is `torch_npu.int4`, `weight` must not be transposed. Otherwise, an error will be raised indicating that the K axes of the `x` matrix and `weight` matrix do not match. In this scenario, you are advised to use non-quantization operators to achieve higher model accuracy and performance.
- When `antiquant_scale` is `torch.float16` or `torch.bfloat16`, the data types of `x` and `antiquant_scale` must be identical in single-operator mode and can differ in graph mode. If the data types differ, the API internally determines whether to convert them to a unified data type. You can dump graph information to inspect the actual data types involved in the computation.

<!-- npu="950" id34 -->
- <term>Ascend 950PR&950DT products</term>:
  - When `x` is `torch.float16` or `torch.bfloat16` and `weight` is `torch.float32`, `torch_npu.float4_e2m1fn_x2`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`, the `antiquant_offset` parameter is not supported.
  - When `x` is `torch.float16` or `torch.bfloat16` and `weight` is `torch.float32`, this is an MX scenario. In this case, the `antiquant_scale` type must be `torch.uint8`, `antiquant_offset` must be None, and `antiquant_group_size` must be 32. For details, see the call example of [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md).
  - When the data type of `weight` is `torch_npu.hifloat8` and it is carried in an 8-bit data type, pass `torch_npu.hifloat8` for `weight_dtype`.
  - When `x` is `torch.float16` or `torch.bfloat16` and `weight` is `torch.int8` in FRACTAL_NZ format, only graph mode is supported. The value ranges of K and N are [2, 65535]. In pergroup scenarios, `x` must be transposed, `weight` must not be transposed, `antiquant_group_size` must be 64 or 128, K must be aligned to `antiquant_group_size`, and N must be aligned to 64.
<!-- end id34 -->

## Examples<a name="en-us_topic_0000001771071862_section14459801435"></a>

- Single-operator call
    <!-- npu="A3,910b" id35 -->
    - When `weight` is not transposed and `quant_scale` is provided, only the following products are supported:
        - <term>Atlas A2 products</term>
        - <term>Atlas A3 products</term>

            ```python
            import torch
            import torch_npu
            # When the input data type is int8, and the layout is ND:
            cpu_x = torch.randn((8192, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(320, 256),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((1, 256),dtype=torch.float16)
            cpu_antiquantoffset = torch.randn((1, 256),dtype=torch.float16)
            cpu_quantscale = torch.randn((1, 256),dtype=torch.float32)
            cpu_quantoffset = torch.randn((1, 256),dtype=torch.float32)
            quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())
            ```
    <!-- end id35 -->
      
    <!-- npu="A3,910b" id36 -->
    - weight transpose+antiquant_scale scenario (`torch.float16`/`torch.bfloat16` type), only the following products are supported:
        - <term>Atlas A2 products</term>
        - <term>Atlas A3 products</term>

            ```python
            import torch
            import torch_npu
            cpu_x = torch.randn((96, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((256),dtype=torch.float16)
            # Construct the `scale` parameter with an `int64` data type.
            antiquant_scale = torch_npu.npu_trans_quant_param(cpu_antiquantscale.to(torch.float32).npu()).reshape(256, 1)
            cpu_antiquantoffset = torch.randint(-128, 127, (256, 1), dtype=torch.int32)
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.transpose(-1,-2).npu(), antiquant_scale.transpose(-1,-2).npu(), cpu_antiquantoffset.transpose(-1,-2).npu())
            ```
    <!-- end id36 -->

    <!-- npu="A3,910b" id37 -->
    - weight transpose+antiquant_scale scenario (where antiquant_scale is of the `torch.float16` type), only the following products are supported:

        - <term>Atlas A2 products</term>
        - <term>Atlas A3 products</term>

            ```python
            import torch
            import torch_npu
            # When the input data type is int8, and the layout is ND:
            cpu_x = torch.randn((96, 320),dtype=torch.float16)
            cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
            cpu_antiquantscale = torch.randn((256,1),dtype=torch.float16)
            cpu_antiquantoffset = torch.randint(-128, 127, (256,1), dtype=torch.float16)
            npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu().transpose(-1, -2), cpu_antiquantscale.npu().transpose(-1, -2), cpu_antiquantoffset.npu().transpose(-1, -2))
            ```
    <!-- end id37 -->

    <!-- npu="950" id41 -->
    - weight transpose+antiquant_scale+`torch_npu.hifloat8` scenario, only <term>Ascend 950PR&950DT products</term> are supported.

        ```python
        import torch
        import torch_npu
        # input int8+ND
        cpu_x = torch.randn((96, 320),dtype=torch.float16)
        # the hifloat8 data type is represented by int8
        cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
        cpu_antiquantscale = torch.randn((256,1),dtype=torch.float16)
        npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu().transpose(-1, -2), cpu_antiquantscale.npu().transpose(-1, -2), weight_dtype = torch_npu.hifloat8)
        ```
    <!-- end id41 -->
    <!-- npu="950" id42 -->
    - FRACTAL_NZ format with the `torch_npu.float4_e2m1fn_x2` type for weight, only <term>Ascend 950PR&950DT products</term> are supported.

        ```python
        import torch
        import torch_npu
        import numpy as np
        from ml_dtypes import float4_e2m1fn
        import math
        m = 1
        k = 64
        n = 64
        antiquant_group_size = 32
        x = torch.randn((m, k),device='npu',dtype=torch.float16)
        cpu_weight = np.random.random(k * n).reshape((k, n)).astype(float4_e2m1fn)
        cpu_weight = torch.from_numpy(cpu_weight.astype(np.float32))
        npu_weight = cpu_weight.npu()
        npu_weight = torch_npu.npu_format_cast(npu_weight, 29, customize_dtype=torch.float16)
        weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
        antiquantscale = torch.randn((math.ceil(k/antiquant_group_size), n),device='npu',dtype=torch.float16)
        npu_out = torch_npu.npu_weight_quant_batchmatmul(x.npu(), weight_packed, antiquantscale,antiquant_group_size = antiquant_group_size)
        ```
    <!-- end id42 -->

- Graph mode call
    - When data layout of `weight` is ND:

        ```python
        # Graph mode
        import torch
        import torch_npu
        import  torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)

        cpu_x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
        cpu_weight = torch.randint(low=-8, high=8, size=(320, 256), dtype=torch.int8, device='npu')
        cpu_antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
        cpu_antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
                return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)

        cpu_model = MyModel()
        model = cpu_model.npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        npu_out = model(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
        ```

    <!-- npu="950" id43 -->
    - ND format with `torch_npu.hifloat8` for the weight input, only <term>Ascend 950PR&950DT products</term> are supported.

        ```python
        # Graph mode
        import torch
        import torch_npu
        import  torchair as tng
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)
        cpu_x = torch.randn((96, 320),dtype=torch.bfloat16)
        cpu_weight = torch.randint(low=-8, high=8, size=(320, 256),dtype=torch.int8)
        cpu_antiquantscale = torch.randn((1, 256),dtype=torch.bfloat16)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
                return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale, quant_offset, bias, antiquant_group_size, weight_dtype=torch_npu.hifloat8)

        cpu_model = MyModel()
        model = cpu_model.npu()
        model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
        npu_out = model(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), None, None, None, None, 0)
        ```
    <!-- end id43 -->
    <!-- npu="310p" id50 -->
    - FRACTAL_NZ format for the weight input, only <term>Atlas inference products</term> are supported.

        ```python
        import torch_npu
        import torch
        from torchair.configs.compiler_config import CompilerConfig
        import torchair as tng
        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)
        class NPUQuantizedLinearA16W8(torch.nn.Module):
            def __init__(self,
                         weight,
                         antiquant_scale,
                         antiquant_offset,
                         quant_offset=None,
                         quant_scale=None,
                         bias=None,
                         transpose_x=False,
                         transpose_weight=True,
                         w4=False):
                super().__init__()

                self.dtype = torch.float16
                self.weight = weight.to(torch.int8).npu()
                self.transpose_weight = transpose_weight

                if self.transpose_weight:
                    self.weight = torch_npu.npu_format_cast(self.weight.contiguous(), 29)
                else:
                    self.weight = torch_npu.npu_format_cast(self.weight.transpose(0, 1).contiguous(), 29) # n,k ->nz

                self.bias = None
                self.antiquant_scale = antiquant_scale
                self.antiquant_offset = antiquant_offset
                self.quant_offset = quant_offset
                self.quant_scale = quant_scale
                self.transpose_x = transpose_x

            def forward(self, x):
                x = torch_npu.npu_weight_quant_batchmatmul(x.transpose(0, 1) if self.transpose_x else x,
                                                           self.weight.transpose(0, 1),
                                                           self.antiquant_scale.transpose(0, 1),
                                                           self.antiquant_offset.transpose(0, 1),
                                                           self.quant_scale,
                                                           self.quant_offset,
                                                           self.bias)
                return x


        m, k, n = 4, 1024, 4096
        cpu_x = torch.randn((m, k),dtype=torch.float16)
        cpu_weight = torch.randint(1, 10, (k, n),dtype=torch.int8)
        cpu_weight = cpu_weight.transpose(-1, -2)

        cpu_antiquantscale = torch.randn((1, n),dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((1, n),dtype=torch.float16)
        cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
        cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)
        model = NPUQuantizedLinearA16W8(cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        out = model(cpu_x.npu())
        ```
    <!-- end id50 -->

    <!-- npu="950,910b" id38 -->
    - FRACTAL_NZ format with `torch.int8` for the weight input, only the following products are supported:

      - <term>Ascend 950PR&950DT products</term>
      - <term>Atlas A2 products</term>

        ```python
        import torch_npu
        import torch
        from torchair.configs.compiler_config import CompilerConfig
        import torchair as tng
        import numpy as np
        config = CompilerConfig()
        config.debug.graph_dump.type = "pbtxt"
        npu_backend = tng.get_npu_backend(compiler_config=config)

        class NPUQuantizedLinearA16W8(torch.nn.Module):
            def __init__(self,
                            weight,
                            antiquant_scale,
                            antiquant_offset,
                            quant_offset=None,
                            quant_scale=None,
                            bias=None,
                            transpose_x=False,
                            transpose_weight=False,
                            w4=False,
                            antiquant_group_size=0):
                super().__init__()
                self.dtype = torch.float16
                self.weight = weight.to(torch.int8).npu()
                self.transpose_weight = transpose_weight
                if self.transpose_weight:
                    self.weight = torch_npu.npu_format_cast(self.weight.transpose(0, 1).contiguous(), 29)
                else:
                    self.weight = torch_npu.npu_format_cast(self.weight.contiguous(), 29)
                self.bias = bias
                self.antiquant_scale = antiquant_scale
                self.antiquant_offset = antiquant_offset
                self.quant_offset = quant_offset
                self.quant_scale = quant_scale
                self.transpose_x = transpose_x
                self.antiquant_group_size = antiquant_group_size

            def forward(self, x):
                x = torch_npu.npu_weight_quant_batchmatmul(x.transpose(0, 1) if self.transpose_x else x,
                                                            self.weight.transpose(0, 1) if self.transpose_weight else self.weight,
                                                            self.antiquant_scale.transpose(0, 1) if self.transpose_weight else self.antiquant_scale,
                                                            self.antiquant_offset.transpose(0, 1) if self.transpose_weight else self.antiquant_offset,
                                                            self.quant_scale,
                                                            self.quant_offset,
                                                            bias=self.bias,
                                                            antiquant_group_size=self.antiquant_group_size)
                return x


        m, k, n = 4, 128, 256
        transpose_weight=False
        group_size = 64


        cpu_x = torch.randn((m, k), dtype=torch.float16)
        cpu_weight = torch.randint(1, 10, (k, n), dtype=torch.int8)
        cpu_bias = torch.randn((1, n), dtype=torch.float16)
        cpu_antiquantscale = torch.randn((k // group_size, n), dtype=torch.float16)
        cpu_antiquantoffset = torch.randn((k // group_size, n), dtype=torch.float16)
        cpu_antiquantscale = cpu_antiquantscale
        cpu_antiquantoffset = cpu_antiquantoffset
        model = NPUQuantizedLinearA16W8(cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),
                                        transpose_weight=transpose_weight, bias=cpu_bias.npu(),
                                        antiquant_group_size=group_size)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        npu_out = model(cpu_x.npu())
        ```
    <!-- end id38 -->
