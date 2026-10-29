# torch\_npu.npu\_grouped\_matmul

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

## Function<a name="en-us_topic_0000002262888689_section1290611593405"></a>

- Provides an efficient method to perform grouped computation of multiple matrix multiplication (MatMul) operations. This API implements batch processing of multiple MatMul operations by grouping MatMul operations with identical or similar shapes, reducing memory access overhead and saving computational resources to improve computational efficiency. For example:

    > [!NOTE]  
    > This operator currently supports non-quantization, full quantization, and fake-quantization scenarios. The full quantization and non-quantization scenarios are applicable to training or inference scenarios, and the fake-quantization scenario is applicable only to inference scenarios.

- Formulas:

    In the following formulas, the $@$ symbol indicates matrix multiplication, and the $\times$ symbol indicates the Hadamard product.

  - Non-quantization scenario (Formula 1):

    $$
    y_i = x_i \text{@} \mathit{weight}_i + \mathit{bias}_i
    $$

  - Static quantization (T-T, T-C) scenarios (Formula 2):

    $$
    y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{offset}_i
    $$

    - `x` is a `torch.int8` input and `bias` is a `torch.int32` input (Formula 2-1):

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i + \mathit{offset}_i
        $$

    - `x` with data type `torch.int8` and `bias` with data type `torch.bfloat16`, `torch.float16`, or `torch.float32`, without `offset` (Formula 2-2):

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{bias}_i
        $$

  - Dynamic quantization (K-T, K-C, T-T, T-C) scenarios (Formula 3):

    $$
    y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
    $$

    - `x` with data type `torch.int8` and `bias` with data type `torch.int32` (Formula 3-1):

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
        $$

    - `x` with data type `torch.int8` and `bias` with data type `torch.bfloat16`, `torch.float16`, or `torch.float32` (Formula 3-2):

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
        $$

  - Fake-quantization scenario (Formula 4):

    $$
    y_i = x_i \text{@} \left((\mathit{weight}_i + \mathit{antiquant\_offset}_i) \times \mathit{antiquant\_scale}_i\right) + \mathit{bias}_i
    $$

  <!-- npu="950" id5 -->
  - Dynamic quantization (G-B, mx) scenarios (Formula 5): applicable only to <term>Ascend 950PR&950DT products</term>

    $$
    \mathit{y}_i[m,n] = \sum_{j=0}^{\mathit{k}_{\mathit{loops}}-1}\left(\left(\sum_{k=0}^{\mathit{gsk}-1}\left(\mathit{x}_{\mathit{slice},i} \times \mathit{weight}_{\mathit{slice},i}\right)\right)\times\left(\mathit{pertokenscale}_i\left[\frac{m}{\mathit{gsm}},j\right] \times \mathit{scale}_i\left[j,\frac{n}{\mathit{gsn}}\right]\right)\right) + \mathit{bias}_i[n]
    $$

    where gsm, gsn, and gsk represent the quantization block sizes of the M, N, and K axes, respectively; K<sub>i</sub> is the K size of each group; x_slice<sub>i</sub> represents the vector of the mth row of x<sub>i</sub> with a length of gsk; weight_slice<sub>i</sub> represents the vector of the nth column of weight<sub>i</sub> with a length of gsk; the K axis is sliced starting from j*gsk, where j is in the range [0, k_loops), and k_loops=ceil(K<sub>i</sub>/gsk); a final slice whose length is less than gsk is supported.
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - Fake-quantization (mx) scenarios (Formula 6): applicable only to <term>Ascend 950PR&950DT products</term>
    - `x` is `torch.float16` or `torch.bfloat16` and `weight` is `torch.float32` (`torch_npu.float4_e2m1fn_x2`) (Formula 6-1):

        $$
        y_i = x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$

    - `x` is `torch.float8_e4m3fn` and `weight` is `torch.float32` (`torch_npu.float4_e2m1fn_x2`) (Formula 6-2):

        $$
        y_i = (x_i \times \mathit{pertokenscale}_i) \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - Fake-quantization (K-CG) scenarios (Formula 7): applicable only to <term>Ascend 950PR&950DT products</term>

    $$
    y_i = \left(x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i)\right) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
    $$

    where antiquant_scale<sub>i</sub> is the pergroup quantization parameter of the weight matrix, scale<sub>i</sub> is the perchannel quantization parameter of the weight matrix, and pertokenscale<sub>i</sub> is the pertoken quantization parameter.
  <!-- end id7 -->

## Prototype<a name="en-us_topic_0000002262888689_section87878612417"></a>

```python
torch_npu.npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=-1, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None, x_dtype=None, weight_dtype=None, scale_dtype=None, per_token_scale_dtype=None)
```

## Parameters<a name="en-us_topic_0000002262888689_section135561610204110"></a>

- **`x`** (`List[Tensor]`): Required. Input matrix list representing the left matrices in MatMul.
  - Valid data types:

    <!-- npu="A3,910b" id8 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `torch.float16`, `torch.float32`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4`.
    <!-- end id8 -->
    <!-- npu="310p" id9 -->
    - <term>Atlas inference products</term>: `torch.float16`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.float16`, `torch.bfloat16`, `torch.float32`, `torch.int8`, `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, `torch.int8`, or `torch_npu.int4`. In practice, `torch_npu.int4` is carried by `torch.int8` or `torch.int32`, and the value range [-8,7] is retained. For `torch_npu.hifloat8` and the float4 series, the optional parameter `x_dtype` must be set to the corresponding type; in this case, the `dtype` of `x` itself no longer takes effect, but the `dtype` of `x` itself must still be an 8-bit data type to ensure the shape is correct. For the float4 series, the inner axis `K` must be an even number to ensure that 8 bits can be converted into two float4 values. In addition, the `torch_npu.float4_e1m2fn_x2` data type is supported only when `weight` is in NZ format.
    <!-- end id10 -->

  - The maximum list length is `128`.

    <!-- npu="950" id11 -->
    - <term>Ascend 950PR&950DT products</term>: In non-quantization scenarios, the maximum list length is 1024.
    <!-- end id11 -->

  - When `split_item` is `0` or `1`, tensors support 2D to 6D inputs in some scenarios. In other cases, only 2D inputs are supported.

- **`weight`** (`List[Tensor]`): Required. Weight matrix list representing the right matrices in MatMul.
  - The data layout supports $ND$/`FRACTAL_NZ`. The supported data types are as follows:

    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
      - When `group_list` is of type `List[int]`, the data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int8`.
      - When `group_list` is of type `Tensor`, the data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`, `torch_npu.int4`, or `torch.int8`.
    <!-- end id12 -->
    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: `torch.float16`.
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT products</term>: `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.int8`, `torch_npu.hifloat8`, `torch.bfloat16`, `torch.float16`, `torch.float32`, `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, `torch.int32`, or `torch_npu.int4`. In practice, `torch_npu.int4` is carried by `torch.int8` or `torch.int32`, and the value range [-8,7] is retained. For `torch_npu.hifloat8` and the float4 series, the optional parameter `weight_dtype` must be set to the corresponding type; in this case, the `dtype` of `weight` itself no longer takes effect, but the `dtype` of `weight` itself must still be an 8-bit data type to ensure the shape is correct. In addition, the `torch_npu.float4_e1m2fn_x2` data type is supported only when `weight` is in NZ format.
      - In full quantization scenarios, when `x` is of the float4 series and `weight` is a float4 series input, only inference scenarios are supported. In this case, the `K` of the input `x` must be an even number, and when `weight` is not transposed, the inner axis `N` must be an even number to ensure that 8 bits can be converted into two float4 values. When the data type of the input `x`/`weight` is `torch.int8` or `torch.float8_e4m3fn` (with `scale_dtype` and `per_token_scale_dtype` being `torch.float8_e8m0fnu`) or `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` (with `scale_dtype` and `per_token_scale_dtype` being `torch.float8_e8m0fnu`), the `weight` data format supports `FRACTAL_NZ`, and the `torch_npu.npu_format_cast` API can be used to convert ND to the `FRACTAL_NZ` format. When `x` is `torch_npu.int4` and `weight` is `torch_npu.int4`, the `K` of the input `x` must also be an even number, and when `weight` is not transposed, the inner axis `N` must be an even number. The `weight` data format supports `FRACTAL_NZ`, and the `torch_npu.npu_format_cast` API can be used to convert ND to the `FRACTAL_NZ` format.
      - In fake-quantization scenarios, when `weight` is of the `torch.float32`/`torch.int32` type, because PyTorch does not natively support some data types, [torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md) can be used to implement **one `torch.float32` value carrying eight `torch_npu.float4_e2m1fn_x2` inputs and one `torch.int32` value carrying eight `torch_npu.int4` inputs**. In this case, `weight_dtype` is not passed. Similarly, when `weight` is `torch_npu.int4`, a pack operation is also required.
      - In fake-quantization scenarios, the MxA8W4 data flow supports `weight` using `torch.uint8` to carry `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` (one `torch.uint8` value carries two `torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` values). In this scenario, static graph mode and dynamic graph mode do not support E=1 or K=64, where E indicates the Group size.
      - In non-quantization scenarios, when the `weight` input is `torch.float16` or `torch.bfloat16` and the last two dimensions of the `weight` shape are aligned to 32B, the data format also supports `FRACTAL_NZ`, and the `torch_npu.npu_format_cast` API can be used to convert ND to the `FRACTAL_NZ` format.
    <!-- end id14 -->

  - The maximum list length is `128`.

    <!-- npu="950" id15 -->
    - <term>Ascend 950PR&950DT products</term>: In non-quantization scenarios, the maximum list length is 1024. In mx quantization full quantization scenarios where `w` is in the `FRACTAL_NZ` format, multiple `weight` tensors can be input, and the maximum tensorlist length is 128.
    <!-- end id15 -->

  - Each input tensor can be 2D or 3D.

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`bias`** (`List[Tensor]`): Optional. Independent bias term for each grouped MatMul output.
  - Valid data types:

    <!-- npu="A3,910b" id16 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `torch.float16`, `torch.float32`, or `torch.int32`.
    <!-- end id16 -->
    <!-- npu="310p" id17 -->
    - <term>Atlas inference products</term>: `torch.float16`.
    <!-- end id17 -->
    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT products</term>:
      - In quantization scenarios, when the input `x` is `torch.int8`, `torch.int32`, `torch.bfloat16`, `torch.float16`, and `torch.float32` are supported; when the input `x` is `torch_npu.float4_e2m1fn_x2`, only `torch.float32` is supported. For other input types, None must be passed.
      - In fake-quantization scenarios, when the input `x` is `torch.float16`, `bias` supports `torch.float16`; when the input `x` is `torch.bfloat16`, `bias` supports `torch.bfloat16` and `torch.float32`; when the input `x` is `torch.float8_e4m3fn` (in MXFP8 and other scenarios), `bias` supports `torch.float16` and `torch.bfloat16`. For other input types, None must be passed.
      - In non-quantization scenarios, when the input `x` is `torch.float16`, `torch.float16` and `torch.float32` are supported; when `x` is `torch.bfloat16`, `torch.bfloat16` and `torch.float32` are supported; when the input `x` is `torch.float32`, `torch.float32` is supported. For other input types, None must be passed.
    <!-- end id18 -->

  - The list length must be identical to that of `weight`.
  - Each input tensor can be 1D or 2D.

- **`scale`** (`List[Tensor]`): Optional. Scaling factor in quantization parameters, used to scale original values to match the quantized range, corresponding to Formula (2) and Formula (3).
  - Valid data types:

    <!-- npu="A3,910b" id19 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Only the $ND$ format is supported.
      - When `group_list` is of type `List[int]`, the data type can be `torch.int64`.
      - When `group_list` is of type `Tensor`, the data type can be `torch.float32`, `torch.bfloat16`, or `torch.int64`.
    <!-- end id19 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas inference products</term>: Only `None` is supported.
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT products</term>:
      - ND format: When the input type of `group_list` is `torch.Tensor` and `x` is not `torch.float16` or `torch.bfloat16`, `torch.int64`, `torch.bfloat16`, `torch.float32`, and `torch.float8_e8m0fnu` are supported. For `torch.float8_e8m0fnu`, `scale_dtype` must be set to the corresponding type; in this case, the `dtype` of `scale` itself no longer takes effect, but the `dtype` of `scale` itself must still be an 8-bit data type to ensure the shape is correct.
      - `FRACTAL_NZ` format: Only `torch.float8_e8m0fnu` is supported, and `scale_dtype` must be set to the corresponding type.
    <!-- end id21 -->

  - The list length must be identical to that of `weight`.

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Each input tensor must be 1D.
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT products</term>: In mx quantization M-axis grouping scenarios, each tensor must be 4D; when `weight` is input as multiple tensors, each tensor must be 3D and have the same transposition attribute as `weight`. In mx quantization K-axis grouping scenarios, each tensor must be 3D and have the same transposition attribute as `weight`. In K-CG fake-quantization M-axis grouping scenarios, each tensor must be 2D and have the same transposition attribute as `weight`.
  <!-- end id23 -->

- **`offset`** (`List[Tensor]`): Optional. Offset in quantization parameters, used to adjust quantized numerical offsets to more accurately represent original floating-point values, corresponding to Formula (2). Currently, only `None` is supported.
  <!-- npu="950" id88 -->
  - <term>Ascend 950PR&950DT products</term>: The S8S4 K-C fake-quantization scenario supports `torch.float32`.
  <!-- end id88 -->
- **`antiquant_scale`** (`List[Tensor]`): Optional. Scaling factor in fake-quantization parameters, used to scale original values to match the fake-quantized range, corresponding to Formula (4).
  - Valid data types:

    <!-- npu="A3,910b" id24 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `torch.float16` or `torch.bfloat16`.
    <!-- end id24 -->
    <!-- npu="310p" id25 -->
    - <term>Atlas inference products</term>: Only `None` is supported.
    <!-- end id25 -->
    <!-- npu="950" id26 -->
    - <term>Ascend 950PR&950DT products</term>: `torch.float16`, `torch.bfloat16`, `torch.float8_e8m0fnu`, where `torch.float8_e8m0fnu` must be passed as `torch.uint8`.
    <!-- end id26 -->

  - The list length must be identical to that of `weight`.
  - Each tensor supports the following shapes, where $g$ indicates the number of MatMul groups, $G$ indicates the number of `pergroup` partitions, and $G_i$ indicates the number of `pergroup` partitions for the $i$-th tensor.
    - In perchannel fake-quantization scenarios, when `weight` is a single tensor, the shape is restricted to [g, n]; when weight is multiple tensors, the shape is restricted to [n<sub>i</sub>].
    - In pergroup fake-quantization scenarios, when `weight` is a single tensor, the shape is restricted to [g, G, n]; when weight is multiple tensors, the shape is restricted to [G<sub>i</sub>, n<sub>i</sub>].
    - In mx fake-quantization scenarios, when `weight` is a single tensor, the shape is restricted to `(g, n, G/2, 2)`

- **`antiquant_offset`** (`List[Tensor]`): Optional. Offset in fake-quantization parameters, used to adjust fake-quantized numerical offsets to more accurately represent original floating-point values, corresponding to Formula (4).
  - Valid data types:

    <!-- npu="A3,910b" id27 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: `torch.float16` or `torch.bfloat16`.
    <!-- end id27 -->
    <!-- npu="310p" id28 -->
    - <term>Atlas inference products</term>: Only `None` is supported.
    <!-- end id28 -->
    <!-- npu="950" id29 -->
    - <term>Ascend 950PR&950DT products</term>: `torch.float16` or `torch.bfloat16`.
    <!-- end id29 -->

  - The list length must be identical to that of `weight`.
  - Each tensor must have an identical input dimension to that of `antiquant_scale`.

- **`per_token_scale`** (`List[Tensor]`): Optional. Scaling factor in quantization parameters, used to scale original values to match the quantized range, representing the scaling factor introduced by `x` quantization in `pertoken` quantization parameters, corresponding to Formula (3).
  - When `group_list` is of type `List[int]`, only `None` can be provided.
  - When `group_list` is of type `Tensor`:

    <!-- npu="A3,910b" id30 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data type can be `torch.float32`.
    <!-- end id30 -->
    <!-- npu="950" id31 -->
    - <term>Ascend 950PR&950DT products</term>: When the `group_list` input type is `Tensor`, the data type supports `torch.float32` and `torch.float8_e8m0fnu`. For `torch.float8_e8m0fnu`, `per_token_scale_dtype` must be set to the corresponding type; in this case, the `dtype` of `per_token_scale` itself no longer takes effect, but the `dtype` of `per_token_scale` itself must still be an 8-bit data type to ensure the shape is correct.
    <!-- end id31 -->
    - The list length must be identical to that of `x`.

    <!-- npu="A3,910b" id32 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Each input tensor must be 1D.
    <!-- end id32 -->
    <!-- npu="950" id33 -->
    - <term>Ascend 950PR&950DT products</term>: In mx quantization, each tensor must be 3D and have the same transposition attribute as `x`. In mx fake-quantization scenarios, the tensor must be 3D with the same transposition attribute as x, and the data type supports `torch.float8_e8m0fnu`. In K-CG fake-quantization scenarios, the tensor must be 1D, and the data type supports `torch.float32`.
    <!-- end id33 -->

- **`group_list`** (`List[int]`/`torch.Tensor`): Optional. Grouping index indicating the index mapping of MatMul along dimension `0` of `x`. The data layout supports $ND$, and the data type supports `torch.int64`.

  <!-- npu="310p" id34 -->
  - <term>Atlas inference products</term>: Only the `torch.Tensor` type is supported. Each input tensor must be 1D, and the length must be identical to that of `weight`.
  <!-- end id34 -->
  <!-- npu="A3,910b" id35 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Both `List[int]` and `torch.Tensor` types are supported. If the type is `torch.Tensor`, each input tensor must be 1D, and the length must be identical to that of `weight`.
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT products</term>: Only the `torch.Tensor` type is supported. Only 1D input is supported. When `group_type` is 0, the length must be the same as the size of the highest axis E of `weight` (a single tensor, [[E, N, K]]). When `group_type` is 2, the length must be the same as the size of the highest axis E of the output (a single tensor, [[E, M, N]]).
  <!-- end id36 -->
  - Configuration requirements:
    - When `group_list` is of type `List[int]`, the configuration must be a non-negative increasing sequence, and the length cannot be 0.
    - When `group_list` is of type `Tensor`:
      - When `group_list_type` is set to `0`, `group_list` must be a non-negative monotonically non-decreasing sequence.
      - When `group_list_type` is set to `1`, `group_list` must be a non-negative sequence, and the length cannot be `1`.
      - When `group_list_type` is set to `2`, the shape of `group_list` must be `[E, 2]`, where `E` indicates the group size. The data layout must be `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`. `groupSize` indicates the size of each group along the grouping axis and must be non-negative.

- **`activation_input`** (`List[Tensor]`): Optional. Backward input of the activation function. Currently, only `None` is supported.
- **`activation_quant_scale`** (`List[Tensor]`): Optional. Reserved parameter. Currently, only `None` is supported.
- **`activation_quant_offset`** (`List[Tensor]`): Optional. Reserved parameter. Currently, only `None` is supported.
- **`split_item`** (`int`): Optional. Specifies the split mode. The data type supports `int64`.
  - `0` or `1`: The output contains multiple tensors, and the number matches that of `weight`.
  - `2` or `3`: The output is a single tensor.

  <!-- npu="950" id37 -->
  - <term>Ascend 950PR&950DT products</term>: In non-quantization and fake-quantization scenarios, the values 0/1/2/3 can be used, and in quantization scenarios, 2 and 3 are supported currently.
  <!-- end id37 -->

- **`group_type`** (`int`): Optional. The grouping axis. The data type supports `int64`.
  - When `group_list` is of type `List[int]`, only `None` is supported.
  - When `group_list` is of type `Tensor`, and the matrix multiplication is $C[m,n]=A[m,k] \times B[k,n]$, the valid enum values are: `-1` (no grouping), `0` (grouping along the m-axis), and `2` (grouping along the k-axis).

    <!-- npu="A3,910b" id38 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, the value can be `-1`, `0`, or `2`.
    <!-- end id38 -->
    <!-- npu="310p" id39 -->
    - <term>Atlas inference products</term>: Currently, only `0` is supported.
    <!-- end id39 -->
    <!-- npu="950" id40 -->
    - <term>Ascend 950PR&950DT products</term>:
      - Non-quantization scenarios support the values -1, 0, and 2. The value 2 is not supported in weightNz scenarios.
      - Quantization scenarios support the values 0 and 2.
      - Fake-quantization scenarios support the values -1 and 0.
    <!-- end id40 -->

- **`group_list_type`** (`int`): Optional. Representation format of `group_list`. The data type supports `int64`.
  - When `group_list` is of type `List[int]`, only `None` is supported.
  - When `group_list` is of type `Tensor`, valid values are `0`, `1`, or `2`.
    - `0`: Default value. The values in `group_list` indicate the cumulative sum (`cumsum`) results of group sizes along the grouping axis.
    - `1`: The values in `group_list` indicate the size of each group along the grouping axis.
    - `2`: The shape of `group_list` must be `[E, 2]`, where `E` indicates the group size. The data layout must be `[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]`. `groupSize` indicates the size of each group along the grouping axis.

    <!-- npu="A3,910b" id41 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The value `2` is supported only when the data type of `x` and `weight` is `torch.int8`, and `group_type` is set to `0` (m-axis grouping).
    <!-- end id41 -->
    <!-- npu="310p" id42 -->
    - <term>Atlas inference products</term>: The value `2` is not supported.
    <!-- end id42 -->
    <!-- npu="950" id43 -->
    - <term>Ascend 950PR&950DT products</term>: The value 2 is supported only in non-quantization, full-quantization, or MX A8W4 fake-quantization scenarios where `group_type` is 0 (m-axis grouping).
    - <term>Ascend 950PR&950DT products</term>: In K-CG fake-quantization scenarios, `group_list_type` supports only the value 1.
    <!-- end id43 -->

- **`act_type`** (`int`): Optional. Activation function type. The data type supports `int64`.
  - When `group_list` is of type `List[int]`, only `None` is supported.
  - When `group_list` is of type `Tensor`, valid enum values are: `0` (no activation), `1` (`RELU` activation), `2` (`GELU_TANH` activation), `3` (not supported), `4` (`FAST_GELU` activation), or `5` (`SILU` activation).

    <!-- npu="A3,910b" id44 -->
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The value ranges from 0 to 5.
    <!-- end id44 -->
    <!-- npu="310p" id45 -->
    - <term>Atlas inference products</term>: Currently, only `0` is supported.
    <!-- end id45 -->
    <!-- npu="950" id46 -->
    - <term>Ascend 950PR&950DT products</term>: The values 0, 1, 2, 4, and 5 are supported currently.
    <!-- end id46 -->

- **`output_dtype`** (`torch.dtype`): Optional. Output data type. Valid values are:
  - `None`: Default value. The output data type is identical to that of `x`.
  - A type identical to that of output `y`. For details, see [Constraints](#en-us_topic_0000002262888689_section618392112366).

- **`tuning_config`** (`List[int]`): Optional. The first element in this list specifies the expected number of tokens processed by each expert. During operator tiling, optimization is performed based on this element for higher performance. For details about the usage scenarios, see [Constraints](#en-us_topic_0000002262888689_section618392112366). Elements starting from the second element are reserved for future extensions and can be omitted. This parameter can be omitted if it is not used.
  <!-- npu="310p" id47 -->
  - <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id47 -->
  <!-- npu="950" id48 -->
  - <term>Ascend 950PR&950DT products</term>: Currently supported only in the S8S4 K-C and S8S4 K-CG fake-quantization scenarios.
    - The first element indicates the expected number of tokens processed by each expert. It supports 0 and positive integers, and the value range is [0, min(M, UINT32_MAX)], where M is the number of rows of the input `x`. The value 0 indicates that no expected value is specified; a positive integer indicates the expected number of tokens processed by each expert and must not be greater than M. Currently, this value is only validated and saved, and does not participate in actual tuning computation.
    - The second element supports only 0 or 1. When it is set to 0 or this element is not passed, the special `weight` format is not enabled, and `weight` is parsed according to the conventional [E, K, N] logical layout. When it is set to 1, the special `weight` format is enabled, and only perchannel scenarios where `offset` is not empty are supported. In this case, `weight` must be arranged as [E, N, K] and then converted to the `FRACTAL_NZ` format, and the `weight` list length must be 1. For example, passing `[0, 1]` indicates that no expected token count is specified and the special `weight` format is enabled.
  <!-- end id48 -->

- **`x_dtype`** (`int`): Optional. Real data type of the input `x`. The default value is None, which means the real data type of the input `x` is the same as the `dtype` of the input `x`.

  <!-- npu="A3,910b,310p" id49 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id49 -->
  <!-- npu="950" id50 -->
  - <term>Ascend 950PR&950DT products</term>: `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2` are supported.
  <!-- end id50 -->

- **`weight_dtype`** (`int`): Optional. Real data type of the input `weight`. The default value is None, which means the real data type of the input `weight` is the same as the `dtype` of the input `weight`. In fake-quantization scenarios where `torch.float32` is used to represent `torch_npu.float4_e2m1fn_x2`, this parameter does not need to be passed.

  <!-- npu="A3,910b,310p" id51 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id51 -->
  <!-- npu="950" id52 -->
  - <term>Ascend 950PR&950DT products</term>: `torch_npu.hifloat8`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2` are supported.
  <!-- end id52 -->

- **`scale_dtype`** (`int`): Optional. Real data type of the input `scale`. The default value is None, which means the real data type of the input `scale` is the same as the `dtype` of the input `scale`.

  <!-- npu="A3,910b,310p" id53 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id53 -->
  <!-- npu="950" id54 -->
  - <term>Ascend 950PR&950DT products</term>: Only `torch.float8_e8m0fnu` is supported currently.
  <!-- end id54 -->

- **`per_token_scale_dtype`** (`int`): Optional. Real data type of the input `per_token_scale`. The default value is None, which means the real data type of `per_token_scale` is the same as the `dtype` of the input `per_token_scale`.

  <!-- npu="A3,910b,310p" id55 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>: This parameter is not supported currently.
  <!-- end id55 -->
  <!-- npu="950" id56 -->
  - <term>Ascend 950PR&950DT products</term>: Only `torch.float8_e8m0fnu` is supported currently.
  <!-- end id56 -->

## Return Values<a name="en-us_topic_0000002262888689_section1558311519405"></a>

**`y`** (`List[Tensor]`): Output tensor. The data format can be ND.

- When `split_item` is set to `0` or `1`, the number of returned tensors is identical to that of `weight`.
- When `split_item` is set to `2` or `3`, a single tensor is returned.

## Constraints<a name="en-us_topic_0000002262888689_section618392112366"></a>

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.

<!-- npu="950" id57 -->
- Constraints on parameter types and formats in graph mode scenarios (applicable only to <term>Ascend 950PR&950DT products</term>):
  - In fake-quantization scenarios, `offset`, `activation_input`, `activation_quant_scale`, and `activation_quant_offset` support only None.
  - In fake-quantization scenarios, `x_dtype` supports only None, and `weight_dtype` supports None or `torch_npu.hifloat8`.
  - In PerGroup quantization scenarios where `x_dtype` is `torch.float16`/`torch.bfloat16` and `weight_dtype` is `torch_npu.int4`, graph mode is not supported.
  - In non-quantization scenarios, `x_dtype` supports only None, and `weight_dtype` supports only None.
  - In non-quantization scenarios, when `x` provides multiple tensors and `y` outputs multiple tensors, `split_item` supports only `0`. When `x` provides a single tensor and `y` outputs multiple tensors, `split_item` supports only `1`. When `y` outputs a single tensor, `split_item` supports `2` or `3`.
  - In full quantization scenarios, if the output data type `y` is `torch.int8`, graph mode is not supported. In full quantization MX scenarios where the `weight` input format is FRACTAL_NZ, if multiple weight tensors are input, graph mode is not supported. If the input types of both `x` and `weight` are `torch_npu.int4`, graph mode is not supported currently.
<!-- end id57 -->
<!-- npu="950" id58 -->
- WeightNZ scenario description (applicable only to <term>Ascend 950PR&950DT products</term>):
  - `weight` supports the FRATCAL_NZ data format only in fake-quantization and full quantization scenarios (where the data types of the input `x`/`weight` are `torch.int8`, or the data types of `x`/`weight` are `torch.float8_e4m3fn`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` with scale_dtype and per_token_scale_dtype being `torch.float8_e8m0fnu`).
  - In full quantization scenarios, when the `weight` input is in the FRATCAL_NZ data format, neither the K axis nor the N axis may be 1. In MXFP4 scenarios, K must be greater than 2, and when weight is not transposed, N must be greater than 2. MXFP4 scenarios support static graph mode but not dynamic graph mode. In MXFP4 static graph mode scenarios, E must be greater than 1 and K must be greater than 64. When both x and weight are `torch_npu.int4`, graph mode is not supported, and K must be kept aligned to 8.
<!-- end id58 -->
<!-- npu="950" id59 -->
- Constraints on the use of the scale NZ affinity format (applicable only to <term>Ascend 950PR&950DT products</term>):
  - Only MX quantization scenarios are supported, where the `scale` data type is `torch.float8_e8m0fnu`.
  - When `scale` uses the NZ affinity format, `weight` must be in the `FRACTAL_NZ` format.
  - The data types of `x` and `weight` support only `torch.float8_e4m3fn`.
  - Only non-transposed `weight` scenarios are supported.
  - Only `group_type` of 0, that is, M-axis grouping, is supported.
  - Graph mode is not supported; only single-operator mode is supported.
  - In single-tensor scenarios, the storage shape of the scale NZ affinity format is $[E, \lceil N/16 \rceil, \lceil K/64 \rceil, 16, 2]$; in multi-tensor scenarios, the storage shape of each scale tensor is $[\lceil N_i/16 \rceil, \lceil K_i/64 \rceil, 16, 2]$.
<!-- end id59 -->
- This API supports graph mode.

  <!-- npu="A3,910b" id60 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The inner dimension limit `InnerLimit` is fixed at `65536`.
  <!-- end id60 -->

- For each tensor group in `x` and `weight`, the size of the last dimension must be less than `InnerLimit`. The last dimension of <code>x<sub>i</sub></code> indicates the K-axis of <code>x<sub>i</sub></code> when `x` is not transposed, or the M-axis of <code>x<sub>i</sub></code> when `x` is transposed. The last dimension of <code>weight<sub>i</sub></code> indicates the $N$ axis of <code>weight<sub>i</sub></code> when `weight` is not transposed, or the $K$ axis of <code>weight<sub>i</sub></code> when `weight` is transposed.

<!-- npu="950" id61 -->
- Usage scenario constraints for `tuning_config`:

    - The S8S4 K-C and S8S4 K-CG fake-quantization scenarios are applicable only to <term>Ascend 950PR&950DT products</term>.
    - Quantization scenarios: The input is `torch.int8`, the output is `torch.int32`/`torch.bfloat16`/`torch.float16`/`torch.int8`, and the scenario is single-tensor single-expert. The data types are as follows.

    | x | weight | output_dtype | y |
    | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int8` | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.float16` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int32` |
<!-- end id61 -->
- Constraints on the input and output data types in different scenarios:

  <!-- npu="A3,910b" id62 -->
  - **When the `group_list` input type is List[int]** (for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>)

    **Table 1**  Data type constraints

    | scenario | x | weight | bias | scale | antiquant_scale | antiquant_offset | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Non-quantization | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | `torch.float16` | `torch.float16` |
    | Non-quantization | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | Non-quantization | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | `torch.float32` | `torch.float32` |
    | perchannelFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | None | None | `torch.int8` | `torch.int8` |
    | fake-quantization | `torch.float16` | `torch.int8` | `torch.float16` | None | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` |
    | fake-quantization | `torch.bfloat16` | `torch.int8` | `torch.float32` | None | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
  <!-- end id62 -->
  <!-- npu="A3,910b" id63 -->
  - **When the `group_list` input type is Tensor** (for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>)

    **Table 2**  Data type constraints

    | scenario | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | Non-quantization | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None | None/`torch.float16` | `torch.float16` |
    | Non-quantization | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | None | None | None | None | None/`torch.bfloat16` | `torch.bfloat16` |
    | Non-quantization | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | None | None/`torch.float32` (only when x/weight/y are all single tensors) | `torch.float32` |
    | perchannelFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | None | None | None | None/`torch.int8` | `torch.int8` |
    | perchannelFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | perchannelFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | None | None | None | `torch.float16` | `torch.float16` |
    | pertokenFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | None | None | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | pertokenFull quantization | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
    | fake-quantization | `torch.float16` | `torch.int8`/`torch_npu.int4` | `torch.float16` | None | `torch.float16` | `torch.float16` | None | None/`torch.float16` | `torch.float16` |
    | fake-quantization | `torch.bfloat16` | `torch.int8`/`torch_npu.int4` | `torch.float32` | None | `torch.bfloat16` | `torch.bfloat16` | None | None/`torch.bfloat16` | `torch.bfloat16` |

    > [!NOTE]
    >
    > - In fake-quantization scenarios, if the type of weight is `torch.int8`, only the perchannel mode is supported; if the type of weight is `torch_npu.int4`, both the perchannel and pergroup modes are supported. If the mode is pergroup, the pergroup count G or G<sub>i</sub> must divide the corresponding k<sub>i</sub>. If weight is multiple tensors, the pergroup length is defined as s<sub>i</sub> = k<sub>i</sub> / G<sub>i</sub>, and all s<sub>i</sub> (i=1,2,...g) must be equal.
    > - In fake-quantization scenarios, if the type of weight is `torch_npu.int4`, the last dimension of each group of tensors in weight must be an even number. The last dimension of weight<sub>i</sub> refers to the N axis of weight<sub>i</sub> when weight is not transposed, or the K axis of weight<sub>i</sub> when weight is transposed. In addition, in pergroup scenarios, when weight is transposed, the pergroup length s<sub>i</sub> must be an even number. Tensor transposition: if the tensor shape is [M,K], the stride is [1,M] and the data layout is [K,M], that is, a non-contiguous tensor.
    > - PyTorch currently does not support the `torch_npu.int4` type. When it is needed, the [torch_npu.npu_quantize](torch_npu-npu_quantize.md) API can be used to represent `torch_npu.int4` with `torch.int32` data.
  <!-- end id63 -->
  <!-- npu="310p" id64 -->
  - **When the `group_list` input type is Tensor** (for <term>Atlas inference products</term>)

    **Table 3**  Data type constraints

    | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
  <!-- end id64 -->
  <!-- npu="950" id65 -->
  - **When the `group_list` input type is Tensor** (for <term>Ascend 950PR&950DT products</term>)

    **Table 4** Data type constraints

    | scenario | x | weight | bias | scale | antiquant_scale | antiquant_offset | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | perchannelfake-quantization | `torch.float16`/`torch.bfloat16` | `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | perchannelfake-quantization | `torch.float16`/`torch.bfloat16` | `torch.int8`/`torch.int32` (`torch_npu.int4`) | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | pergroupfake-quantization | `torch.float16`/`torch.bfloat16` | `torch.int32`(`torch_npu.int4`) | `torch.float16`/`torch.float32`/None | None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | mxfake-quantization | `torch.float16`/`torch.bfloat16` | `torch.float32` (`torch_npu.float4_e2m1fn_x2`) | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | None | `torch.float8_e8m0fnu` | None | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | mxfake-quantization | `torch.float8_e4m3fn` | `torch.uint8`/`torch.int8` (`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`) | `torch.float16`/`torch.bfloat16` | None | `torch.float8_e8m0fnu` | None | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | K-CGfake-quantization | `torch.int8` | `torch.int32` (`torch_npu.int4`) | `torch.float32` | `torch.float32` | `torch.float16` | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | MxA8W8 (Full quantization) | NDsupports `torch.float8_e4m3fn`/`torch.float8_e5m2`, FRACTAL_NZsupports `torch.float8_e4m3fn` | NDsupports `torch.float8_e4m3fn`/`torch.float8_e5m2`, <br>FRACTAL_NZsupports `torch.float8_e4m3fn` | None | `torch.float8_e8m0fnu` | None | None | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | MxA4W4 (Full quantization) | NDsupports `torch_npu.float4_e2m1fn_x2`, FRACTAL_NZsupports `torch_npu.float4_e1m2fn_x2` | NDsupports `torch_npu.float4_e2m1fn_x2`, FRACTAL_NZsupports `torch_npu.float4_e1m2fn_x2` | `torch.float32`/None | `torch.float8_e8m0fnu` | None | None | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-T && T-C static quantization | `torch.int8` | `torch.int8` | `torch.int32`/None | `torch.int64` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.int8` | `torch.float16`/`torch.bfloat16`/`torch.int8` |
    | T-T && T-C static quantization | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | T-T && T-C static quantization | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | None | None | None | `torch.float16` | `torch.float16` |
    | T-T && T-C static quantization | `torch.int8` | `torch.int8` | `torch.int32`/None | None/`torch.int64` | None | None | None | `torch.int32` | `torch.int32` |
    | T-T && T-C static quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.int64`/`torch.float32` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-T && T-C static quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.int64`/`torch.float32` | None | None | None | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-T && K-C dynamic quantization | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | None | None | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | K-T && K-C dynamic quantization | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | None | None | `torch.float32` | `torch.float16` | `torch.float16` |
    | K-T && K-C dynamic quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-T && K-C dynamic quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-C dynamic quantization | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | K-G quantization | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | T-C && T-T dynamic quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-C && T-T dynamic quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B dynamic quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B dynamic quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | Non-quantization | `torch.float16` | `torch.float16` | `torch.float16`/`torch.float32` | None | None | None | None | `torch.float16` | `torch.float16` |
    | Non-quantization | `torch.bfloat16` | `torch.bfloat16` | `torch.float16`/`torch.float32` | None | None | None | None | `torch.bfloat16` | `torch.bfloat16` |
    | Non-quantization | `torch.float32` | `torch.float32` | `torch.float32` | None | None | None | None | `torch.float32` | `torch.float32` |
    | S8S4 K-C/K-CGfake-quantization | `torch.int8` | `torch_npu.int4` | `torch.float32` | `torch.uint64` | None | None | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
  <!-- end id65 -->

- The following scenarios are supported based on the variations in the tensor counts of input `x`, input `weight`, and output `y`. In these scenario descriptions, "single" indicates a single tensor, and "multiple" indicates multiple tensors. The naming sequence follows the order of `x`, `weight`, and `y`. For example, "single-multiple-single" indicates that `x` is a single tensor, `weight` is a list of tensors, and `y` is a single tensor.

  <!-- npu="A3,910b" id66 -->
  - **Scenarios supported when the `group_list` input type is List[int]** (for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>)

    | Supported Scenario | Scenario Description | Scenario Restrictions |
    | --- | --- | --- |
    |Multiple-Multiple-Multiple|`x`, `weight`, and `y` are lists of tensors. The tensors of each data group are independent.|1. `split_item` must be `0` or `1`.<br>2. The tensors in `x` must be identical in dimension and can be 2D to 6D. The tensors in `weight` must be 2D. The tensor dimensions in `y` must be identical to those of `x`.<br>3. When the tensors in `x` are greater than 2D, `group_list` must be omitted.<br>4. When `x` contains 2D tensors and `group_list` is provided, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`.|
    |Single-Multiple-Single|`x` and `y` are single tensors, and `weight` is a list of tensors.|1. `split_item` must be `2` or `3`.<br>2. `group_list` must be provided, and its last value must be identical to the first dimension of the tensor in `x`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.<br>4. The N-axis size of each tensor in `weight` must be identical.|
    |Single–Multiple–Multiple|`x` is a single tensor, and `weight` and `y` are lists of tensors.|1. `split_item` must be `0` or `1`.<br>2. `group_list` must be provided, and the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `y`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.|
    |Multiple–Multiple–Single|`x` and `weight` are lists of tensors, and `y` is a single tensor. The results of each matrix multiplication group are stored sequentially within the same continuous tensor.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `x`, `weight`, and `y` must be 2D.<br>3. The N-axis size of each tensor in `weight` must be identical.<br>4. If `group_list` is provided, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`.|
  <!-- end id66 -->
  <!-- npu="A3,910b" id67 -->
  - **Scenarios supported when the `group_list` input type is Tensor** (for <term>Atlas A2 products</term> and <term>Atlas A3 products</term>)

    > [!NOTE]   
    >
    > - Quantization and fake-quantization are supported only when `group_type` is set to `-1` or `0`.
    > - Activation function computation is supported only in `pertoken` quantization scenarios.

    |group_type|Supported Scenario|Description|Constraints|
    | --- | --- | --- | --- |
    |-1|Multiple-Multiple-Multiple|`x`, `weight`, and `y` are lists of tensors. The tensors of each data group are independent.|1. `split_item` must be `0` or `1`.<br>2. The tensors in `x` must be identical in dimension and can be 2D to 6D. The tensors in `weight` must be 2D. The tensor dimensions in `y` must be identical to those of `x`.<br>3. `group_list` must be omitted.<br>4. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>5. `x` does not support transposition.|
    |0|Single–Single–Single|`x`, `weight`, and `y` are all single tensors.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>3. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `2`, the sum of the values in its second column must be identical to the first dimension of the tensor in `x`.<br>4. The maximum size of the first dimension of `group_list` is `1024`. That is, a maximum of `1024` groups are supported.<br>5. Transposition of `weight` is supported.<br>6. `x` does not support transposition.|
    |0|Single-Multiple-Single|`x` and `y` are single tensors, and `weight` is a list of tensors.|1. `split_item` must be `2` or `3`.<br>2. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`, and the maximum length is `128`. When `group_list_type` is set to `2`, the sum of the values in its second column must be identical to the first dimension of the tensor in `x`, and the maximum length is `128`.<br>3. The tensors in `x`, `weight`, and `y` must be 2D.<br>4. The N-axis size of each tensor in `weight` must be identical.<br>5. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>6. `x` does not support transposition.|
    |0|Multiple–Multiple–Single|`x` and `weight` are lists of tensors, and `y` is a single tensor. The results of each matrix multiplication group are stored sequentially within the same continuous tensor.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `x`, `weight`, and `y` must be 2D.<br>3. The N-axis size of each tensor in `weight` must be identical.<br>4. If `group_list` is provided, when `group_list_type` is set to `0`, the differences between adjacent elements in `group_list` must correspond one-to-one to the first dimension of each tensor in `x`. When `group_list_type` is set to `1`, the values of `group_list` must correspond one-to-one to the first dimension of each tensor in `x`, and the maximum length is `128`. When `group_list_type` is set to `2`, the values in the second column of `group_list` must correspond one-to-one to the first dimension of each tensor in `x`, and the maximum length is `128`.<br>5. Transposition of `weight` is supported, but the transpose configuration of each tensor in `weight` must be identical.<br>6. `x` does not support transposition.|
    | 2 | single-single-single | x, weight, and y are all single tensors | 1. Only split_item of 2 or 3 is supported.<br>  2. The tensors in x and weight must be 2D, and the tensor in y must be 3D.<br>  3. group_list must be passed. When group_list_type is 0, the last value must be equal to the second dimension of the tensor in x; when group_list_type is 1, the sum of the values must be equal to the second dimension of the tensor in x; when group_list_type is 2, the sum of the values in the second column must be less than or equal to the second dimension of the tensor in x.<br>  4. The maximum length of the first dimension of group_list is 1024, that is, a maximum of 1024 groups is supported.<br>  5. x must be transposed, and weight must not be transposed. |
  <!-- end id67 -->
  <!-- npu="310p" id68 -->
  - **Scenarios supported when the `group_list` input type is Tensor** (for <term>Atlas inference products</term>)

    The inputs and outputs support only the `torch.float16` data type. The N-axis size of the output `y` must be a multiple of `16`.

    |group_type|Supported Scenario|Description|Constraints|
    | --- | --- | --- | --- |
    |0|Single-Single-Single|`x`, `weight`, and `y` are all single tensors.|1. `split_item` must be `2` or `3`.<br>2. The tensors in `weight` must be 3D, and the tensors in `x` and `y` must be 2D.<br>3. `group_list` must be provided. When `group_list_type` is set to `0`, its last value must be identical to the first dimension of the tensor in `x`. When `group_list_type` is set to `1`, the sum of its values must be identical to the first dimension of the tensor in `x`.<br>4. The maximum size of the first dimension of `group_list` is `1024`. That is, a maximum of `1024` groups are supported.<br>5. Transposition of `weight` is supported, but `x` does not support transposition.|
  <!-- end id68 -->
  <!-- npu="950" id69 -->
  - **Scenarios supported when the `group_list` input type is Tensor** (<term>Ascend 950PR&950DT products</term>)

    > [!NOTE]
    >
    > - Full quantization supports only single-single-single scenarios with group_type of 0 and 2. Training scenarios, that is, group_type of 2, are supported only in full quantization where the input x and weight are `torch_npu.hifloat8`, `torch.float8_e4m3fn`, or `torch.float8_e5m2`.
    > - Fake-quantization scenarios support multi-multi-multi scenarios with `group_type` of -1, and single-single-single and single-multi-single scenarios with `group_type` of 0. Single-multi-single indicates that `x` is a single tensor, `weight` is multiple tensors, and `y` is a single tensor.
    >   - perchannel fake-quantization scenarios where the input `x` is `torch.float16`/`torch.bfloat16` and `weight` is `torch.int8`/`torch.int32` (`torch_npu.int4`).
    >   - pergroup fake-quantization scenarios where the input `x` is `torch.float16`/`torch.bfloat16` and `weight` is `torch.int32` (`torch_npu.int4`).
    > - Fake-quantization scenarios do not support activation computation. In full quantization scenarios, when x and weight are `torch.int8`, the quantization mode is static T-C quantization or dynamic K-C quantization, and the scale type is `torch.float32`/`torch.bfloat16`, the act_type parameter supports the values 0, 1, 2, 4, and 5. In other full quantization scenarios, the act_type parameter supports only the value 0.
    > - In non-quantization scenarios where x, weight, and out use `torch.float16`/`torch.bfloat16`, the data format is ND, and `group_type` is 0/-1, `act_type` supports `0`/`2`. In all other scenarios, `act_type` supports only `0`.

    | group_type | Supported Scenario | Scenario Description | Scenario Restrictions |
    | --- | --- | --- | --- |
    | -1 | multi-multi-multi | x and weight are multiple tensors, and y is multiple tensors. The tensors of each group of data are independent. | 1. Only split_item of 0 or 1 is supported (non-quantization supports only split_item of 0).<br>  2. The tensors in x must have consistent dimensions. In pergroup fake-quantization scenarios, only 2D is supported; other scenarios support 2 to 6 dimensions, and non-quantization supports 2D. The tensors in weight must be 2D. The tensor dimensions in y must be consistent with those in x.<br>  3. `group_list` must be passed as empty.<br>  4. weight transposition is supported, but whether each tensor in weight is transposed must be uniform.<br>  5. x does not support transposition.<br>  6. In non-quantization and fake-quantization scenarios, bias is optional and the shape supports only the 1D tensorList[(n),(n),...,(n)], and the list length is the same as the weight length.<br>  7. In pergroup fake-quantization scenarios, the last dimension of weight must be an even number, and K must be an integer multiple of group_size; in multi-multi-multi scenarios, non-transposed weight is not supported. |
    | 0 | single-single-single | `x`, `weight`, and `y` are all single tensors. | 1. Only `split_item` of 2 or 3 is supported.<br>  2. `group_list` must be passed. When `group_list_type` is 0, the last value must not be greater than the first dimension of the tensor in `x`; when `group_list_type` is 1, the sum of the values must not be greater than the first dimension of the tensor in `x`. When `group_list_type` is 2, the sum of the values in the second column must be less than or equal to the first dimension of the tensor in `x`.<br>  3. The maximum length of the first dimension of `group_list` is 1024, that is, a maximum of 1024 groups is supported.<br>  4. In quantization scenarios, `bias` is supported only when the input `dtype` of `x` is `torch.int8`/`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`; both non-quantization and fake-quantization scenarios support `bias`.<br>  5. `x` supports only non-transposed input, and `weight` supports transposed or non-transposed input. However, when both `x` and `weight` are `torch_npu.int4` and the `weight` format is ND, only non-transposed input is supported.<br>  6. A single tensor in `weight` must be 3D, and a single tensor in `x` and `y` must be 2D.<br>  7. The K axis must not be 0.<br>  8. In mx fake-quantization scenarios where the input `dtype` of `x` is `torch.float16`/`torch.bfloat16` and `weight` is `torch.float32` (`torch_npu.float4_e2m1fn_x2`), only non-transposed `x` and non-transposed `weight` are supported. In perchannel fake-quantization scenarios where the input `dtype` of `x` is `torch.float16`/`torch.bfloat16` and `weight` is `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8`, and in mx fake-quantization scenarios where the input `dtype` of `x` is `torch.float8_e4m3fn` and `weight` is `torch.float32` (`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`), only non-transposed `x` and transposed `weight` are supported.<br>  9. In mx fake-quantization and K-CG fake-quantization scenarios, the last two dimensions of `weight` must be aligned to 32B.<br>  10. In mx fake-quantization scenarios where the input `dtype` of `x` is `torch.float8_e4m3fn` and `weight` is `torch.float32` (`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`), the type of `bias` must be the same as the `output_dtype` data type, and the optional parameter `output_dtype` must be passed.<br>  11. `bias` is optional and the shape supports only the 2D (g,n), where g is the number of groups of M. |
    | 0 | single-multi-single | `x` is a single tensor, `weight` is multiple tensors, and `y` is a single tensor. | 1. Only `split_item` of 2 or 3 is supported.<br>  2. `group_list` must be passed. When `group_list_type` is 0, the last value must be equal to the first dimension of the tensor in `x`; when `group_list_type` is 1, the sum of the values must correspond one-to-one to the first dimension of the tensor in `x`, and the maximum length is 128 (1024 in non-quantization scenarios).<br>  3. The tensors in `x` and `y` must be 2D, with shapes (M, K) and (M, N), respectively.<br>  4. The tensors in `weight` must be 2D, with shapes (N, K) or (K, N), respectively.<br>  5. The N axis of each tensor in `weight` must be equal.<br>  6. `weight` transposition is supported, but whether each tensor in `weight` is transposed must be uniform.<br>  7. `x` does not support transposition.<br>  8. Non-quantization and fake-quantization scenarios support `bias`. `bias` is optional and the shape supports only the 1D tensorList[(n),(n),...,(n)], and the list length is the same as the `weight` list length.<br>  9. In full quantization MX scenarios where the input `w` is in the `FRACTAL_NZ` format, single-multi-single scenarios are supported, and `bias` is not supported in this case. |
    | 0 | multi-multi-single | x and weight are multiple tensors, and y is a single tensor. The result of each group of matrix multiplication is stored contiguously in the same tensor. | 1. Only split_item of 2 or 3 is supported.<br>  2. The tensors in x, weight, and y must be 2D.<br>  3. The N axis of each tensor in weight must be equal.<br>  4. If group_list is passed, when group_list_type is 0, the differences in group_list must correspond one-to-one to the first dimension of the tensor in x; when group_list_type is 1, the values in group_list must correspond one-to-one to the first dimension of the tensor in x, and the maximum length is 128 (1024 in non-quantization scenarios).<br>  5. weight transposition is supported, but whether each tensor in weight is transposed must be uniform.<br>  6. x does not support transposition.<br>  7. In non-quantization scenarios, bias is optional and the shape supports only the 1D tensorList[(n),(n),...,(n)], and the list length is the same as the weight length. |
    | 2 | single-single-single | x, weight, and y are all single tensors. | 1. Only split_item of 2 or 3 is supported.<br>  2. group_list must be passed. When group_list_type is 0, the last value must not be greater than the first dimension of the tensor in x; when group_list_type is 1, the sum of the values must not be greater than the first dimension of the tensor in x.<br>  3. The maximum length of the first dimension of group_list is 1024, that is, a maximum of 1024 groups is supported.<br>  4. In full quantization scenarios, scale of `torch.int64` is not supported.<br>  5. Only transposed x and non-transposed weight are supported.<br>  6. A single tensor in y must be 3D, and a single tensor in x and weight must be 2D.<br>  7. In graph mode scenarios, the K axis must not be 0.<br>  8. Only ND input and ND output are supported. |
    | 2 | single-multi-multi | x is a single tensor, weight is multiple tensors, and y is multiple tensors. | 1. Only split_item of 0 or 1 is supported.<br>  2. The tensors in x, weight, and y must be 2D.<br>  3. If grouplist is passed, when group_list_type is 0, the differences in group_list must correspond one-to-one to the first dimension of the tensor in x; when group_list_type is 1, the values in group_list must correspond one-to-one to the first dimension of the tensor in x, and the maximum length is 1024.<br>  4. Only transposed x and non-transposed weight are supported.<br>  5. The shape of the non-grouping axis must not be 0.<br>  6. Only ND input and ND output are supported.<br>  7. bias is not supported.<br>  8. Only non-quantization is supported. |

    - **Single-single-single full quantization scenarios**: The shape restrictions on scale and per_token_scale in each scenario are as follows (where g is the shape size of `group_list`, M is the same as the shape m of x, K is the same as the shape k of x, and N is the same as the shape n of weight).

        | group_type | Quantization Mode | Shape Restriction |
        | --- | --- | --- |
        | 0/2 | T-T && T-C static quantization | <li>scale is a single tensor, and each tensor is 2D or 1D: in C quantization scenarios, the shape is (g, N); in T quantization scenarios, the shape is (g, 1) or (g,). In `torch.int8` output scenarios, only C quantization is supported, with the shape (g, N).</li><li>per_token_scale does not need to be assigned.</li> |
        | 0/2 | K-T && K-C dynamic quantization | <li>scale is a single tensor, and each tensor is 2D or 1D: in C quantization scenarios, the shape is (g, N); in T quantization scenarios, the shape is (g, 1) or (g,).</li><li>per_token_scale is a single tensor. When group_type is 0, the shape is 1D (M,); when group_type is 2, the shape is 2D (g, M).</li> |
        | 0/2 | T-T dynamic quantization | Both scale and per_token_scale are a single tensor. When 2D, the shape is (g, 1); when 1D, the shape is (g, ). |
        | 0/2 | T-C dynamic quantization | <li>scale is a single tensor, and each tensor is 2D with the shape (g, N).</li><li>per_token_scale is a single tensor, and each tensor is 2D or 1D: when 2D, the shape is (g, 1); when 1D, the shape is (g, ).</li> |
        | 0 | mx quantization | In Formula (4), gsm = gsn = 1 and gsk = 32. Both scale and per_token_scale are a single tensor. The shape of scale supports (g, N, ceil(K/64), 2) or (g, ceil(K/64), N, 2), and the shape of per_token_scale supports only (M, ceil(K/64), 2). When the input x is `torch_npu.float4_e2m1fn_x2`, K must be an even number and K must not be 2. When weight is not transposed, N must also be an even number. |
        | 2 | mx quantization | In Formula (4), gsm = gsn = 1 and gsk = 32. Both scale and per_token_scale are a single tensor. The shape of scale supports only (K//64+g, N, 2), and the shape of per_token_scale supports only (K//64 + g, M, 2). |
        | 0 | G-B dynamic quantization | In Formula (4), gsm = 1 and gsn = gsk = 128. scale is a single tensor with 3 dimensions and the shape (g, ceil(K/128), ceil(N/128)) or (g, ceil(N/128), ceil(K/128)).<br>per_token_scale is a single tensor with 2 dimensions and the shape (M,ceil(K/128)). |
        | 2 | G-B dynamic quantization | In Formula (4), gsm = 1 and gsn = gsk = 128. scale is a single tensor with 2 dimensions and the shape (K//128+g, ceil(N/128)).<br>per_token_scale is a single tensor with 2 dimensions and the shape (K//128+g, M) |
        | 0 | K-G quantization | <li>Each tensor is 3D with the shape (E, G, N). $G$ must divide $K$, and $k/G$ must be an even number.</li><li>per_token_scale is a single tensor with the 1D shape (M,).</li> |

        > **Single-single-single full quantization special-case description**:
        > - When group_type is 0 or 2, N=1, and the shape of scale is `(g, 1)`, and weight can be either pertensor quantized or perchannel quantized, the pertensor quantization mode is preferred.
        > - When group_type is 2, M=1, and the shape of per_token_scale is `(g, 1)`, and x can be either pertoken quantized or pertensor quantized, the pertensor quantization mode is preferred.
        > - In dynamic quantization scenarios, when group_type is 2, K<128, N<=128, and the shape of scale is 2D `(g, 1)`, according to the existing quantization mode distinction rules, the mode can be either non-G-B dynamic quantization or G-B dynamic quantization. Such scenarios are now uniformly treated as G-B dynamic quantization. If non-G-B dynamic quantization is expected, scale is recommended to be 1D `(g,)` to avoid confusion with G-B dynamic quantization.
        > - In dynamic quantization scenarios, when group_type is 0, g = M, K > 128, and the shape of per_token_scale is `(g,)`, the pertoken quantization mode is selected for x. When group_type is 0, g = M, K <= 128, and the shape of per_token_scale is `(g, 1)`, the quantization mode of x is selected according to the quantization mode of weight (if weight is perchannel or pertensor quantized, pertensor quantization is selected for x; if weight is perblock quantized, pergroup quantization is selected for x).
        > - In dynamic quantization scenarios, when group_type is 2, K<128, and M is not 1, if N is less than or equal to 128, pergroup quantization is selected for x; if N is greater than 128, the quantization mode of x is selected according to the quantization mode of weight (if weight is perchannel or pertensor quantized, pertoken quantization is selected for x; if weight is perblock quantized, pergroup quantization is selected for x).
        > - In dynamic quantization scenarios, when group_type is 2, K<128, M is 1, and the shape of per_token_scale is `(g, 1)`, if N is less than or equal to 128, pergroup quantization is selected for x; if N is greater than 128, the quantization mode of x is selected according to the quantization mode of weight (if weight is perchannel or pertensor quantized, pertensor quantization is selected for x; if weight is perblock quantized, pergroup quantization is selected for x).
        > - In full quantization `torch_npu.int4` scenarios where x is pertoken quantized and weight is pergroup quantized, the pergroup count G must divide K, and K/G must be an even number. After weightNZ transposition, K/G must be aligned to 64, K must be aligned to 64, and N must be aligned to 16.

    - **Fake-quantization scenarios**: The shape restrictions on antiquant_scale, scale, and per_token_scale in each scenario are as follows (where g is the shape size of `group_list`, M and K are the same as the m and k of x, and N is the same as the n of weight).

        | group_type | Quantization Mode | Shape Restriction |
        | --- | --- | --- |
        | -1 | perchannel | antiquant_scale is multiple tensors, and each tensor is 1D with the shape (ni). |
        | 0 | perchannel | antiquant_scale is a single tensor, and each tensor is 2D with the shape (g, N). |
        | -1 | pergroup | <li>group_size supports 32/64/128/256, the pergroup count G=k/gs, and K must be divisible by gs.</li><li>antiquant_scale/antiquant_offset are multiple tensors, and each tensor is 2D with the shape (Gi, ni) and is fixed as non-transposed.</li> |
        | 0 | pergroup | <li>group_size supports 32/64/128/256, the pergroup count G=k/gs, and K must be divisible by gs.</li><li>antiquant_scale/antiquant_offset are a single tensor, and each tensor is 3D with the shape (g, G, N) and is fixed as non-transposed.</li> |
         | 0 | mx | The pergroup count G=K/32, and K must be divisible by 32.<br><ul><li>Scenarios where `x` is a `torch.float16` or `torch.bfloat16` input and `weight` is a `torch.float32` (`torch_npu.float4_e2m1fn_x2`) input: `antiquant_scale` is a single tensor, and each tensor is 3D with the shape (g, K/32, N).</li><li>Scenarios where `x` is a `torch.float8_e4m3fn` input and `weight` is a `torch.uint8`/`torch.int8` (`torch_npu.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`) input: <ul><li>G must be an even number.</li><li>single-single-single: `antiquant_scale` is a single tensor, and each tensor is 4D with the shape (g, N, K/64, 2).</li><li>single-single-single: `per_token_scale` is a single tensor, and each tensor is 3D with the shape (M, K/64, 2).</li><li>single-multi-single: `antiquant_scale` is a single tensor, and each tensor is 3D with the shape (N, K/64, 2).</li><li>single-multi-single: `per_token_scale` is a single tensor, and each tensor is 3D with the shape (M, K/64, 2).</li></ul></li></ul> |
        | 0 | K-CG | <li>Only `group_list_type` of 1 is supported. In this case, the values in `group_list` indicate the size of each group on the grouping axis.</li><li>antiquant_scale is a single tensor, and each tensor is 3D with the shape (g, G, N), where the pergroup count G=K/gs. gs supports the values 128, 192, 256, and 512, and K must be divisible by gs.</li><li>per_token_scale is a single tensor, and each tensor is 1D with the shape (M).</li><li>scale is a single tensor, and each tensor is 2D with the shape (g, N).</li> |
        | 0 | S8S4 K-CG | scale is a single tensor with the shape (g, K/256, N); per_token_scale is a single tensor with the shape (M). |
        | 0 | S8S4 K-C | Both scale and offset are a single tensor with the shape (g, 1, N); per_token_scale is a single tensor with the shape (M). |
  <!-- end id69 -->

## Examples<a name="en-us_topic_0000002262888689_section1566973054111"></a>

- Single-operator call

  - Non-quantization scenario example:

    ```python
    import torch
    import torch_npu
    M = 256
    K = 256
    N = 256
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)
    group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```

  <!-- npu="950" id70 -->
  - <term>Ascend 950PR&950DT products</term>: `torch_npu.hifloat8` input, K-T dynamic quantization scenario

    ```python
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, 1), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8, group_list_type=0)
    ```
  <!-- end id70 -->
  <!-- npu="950" id71 -->
  - <term>Ascend 950PR&950DT products</term>: mx quantization scenario example - mxfp8

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, N, math.ceil(K/64), 2), dtype=torch.int8).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/64), 2), dtype=torch.int8).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id71 -->
  <!-- npu="950" id72 -->
  - <term>Ascend 950PR&950DT products</term>: G-B dynamic quantization scenario example

    ```python
    # G-B
    import math
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id72 -->
  <!-- npu="950" id73 -->
  - <term>Ascend 950PR&950DT products</term>: full quantization weightNZ scenario example (x/weight are `torch.int8`)

    ```python
    import torch
    import torch_npu
    M = 4
    K = 4
    N = 5
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (g, K, N), dtype=torch.int8).npu()
    weightnz = torch_npu.npu_format_cast(weight1, 29)
    scale = torch.randint(-1, 1, (g, N), dtype=torch.bfloat16).npu()
    pertoken = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (g, N), dtype=torch.int32).npu()

    npu_out = torch_npu.npu_grouped_matmul([x], [weightnz], scale=[scale], bias = [bias], per_token_scale = [pertoken], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id73 -->
  <!-- npu="950" id74 -->
  - <term>Ascend 950PR&950DT products</term>: non-quantization weightNZ scenario example

    ```python
    import torch
    import torch_npu
    M = 1
    K = 256
    N = 256
    g = 1
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    weight_flag = 29
    weight = torch_npu.npu_format_cast(weight, weight_flag)
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```
  <!-- end id74 -->
  <!-- npu="950" id75 -->
  - <term>Ascend 950PR&950DT products</term>: mx (A4W4) full quantization scenario example - mxfp4

    ```python
    import math
    import torch
    import torch_npu
    M = 64
    N = 128
    K = 128
    g = 1
    # Use uint8 in place of two mxfp4 values. In this case, the inner axis of uint8 must be expanded by a factor of 2 to match the real mxfp4 inner axis.
    x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()

    weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()

    scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
    per_token_scale = torch.randint(-1, 1, (M, math.ceil(K/64),2), dtype=torch.int8).npu()

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
    split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.float4_e2m1fn_x2, weight_dtype=torch_npu.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id75 -->
  <!-- npu="950" id76 -->
  - <term>Ascend 950PR&950DT products</term>: (A16W4) fake-quantization scenario example

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list, group_list_type):
            for i in range(len(weight)):
                weight[i] = weight[i].transpose(-2, -1)
            return torch_npu.npu_grouped_matmul(x, weight, bias=bias, antiquant_scale=antiquant_scale,
                                                antiquant_offset=antiquant_offset, group_list=group_list,
                                                split_item=split_item, group_type=group_type, output_dtype=torch.float16,
                                                group_list_type=group_list_type)

    def main():
        g, m, k, n = 2, 5, 64, 128
        x1 = torch.ones(m, k, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, n, k, dtype=torch.int32)
        weight_packed = torch_npu.npu_convert_weight_to_int4pack(weight1.npu())
        weight = [weight_packed.npu()]
        bias1 = torch.ones(g, n, dtype=torch.float16)
        bias = [bias1.npu()]
        antiquant_scale1 = torch.ones(g, n, dtype=torch.float16)
        print(f"antiquant_scale1 shape {antiquant_scale1.shape}")
        antiquant_scale = [antiquant_scale1.npu()]
        antiquant_offset1 = torch.zeros(g, n, dtype=torch.float16)
        antiquant_offset = [antiquant_offset1.npu()]
        group_list = torch.Tensor([2, 3]).to(torch.int64).npu()
        group_list_type = 1
        split_item = 3
        group_type = 0
        custom_output = call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list,
                              group_list_type)

    if __name__ == '__main__':
        main()
    ```
  <!-- end id76 -->
  <!-- npu="950" id77 -->
  - <term>Ascend 950PR&950DT products</term>: mx fake-quantization scenario example (x is `torch.float8_e4m3fn`, weight uses `torch.uint8` to carry `torch_npu.float4_e2m1fn_x2` (one `torch.uint8` value carries two float4_e2m1 values), and antiquant_scale and per_token_scale use `torch.uint8` to carry `torch.float8_e8m0fnu`)

    ```python
    import torch
    import torch_npu

    MX_GROUP_SIZE = 32

    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type,
                    output_dtype=torch.bfloat16):
            weight = weight.transpose(-2, -1)
            antiquant_scale = antiquant_scale.transpose(-3, -2)
            bias = [bias] if bias is not None else None
            output = torch_npu.npu_grouped_matmul([x], [weight], bias=bias, group_list=group_list, group_type=0,
                                                  group_list_type=group_list_type, per_token_scale=[per_token_scale],
                                                  antiquant_scale=[antiquant_scale], split_item=2,
                                                  output_dtype=output_dtype, per_token_scale_dtype=torch_npu.float8_e8m0fnu, weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output

    def main():
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        x = torch.ones((m, k)).to(torch.float8_e4m3fn).npu()
        weight = torch.randint(0, 256, (g, n, k//2), dtype=torch.uint8).npu()
        weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)
        per_token_scale = torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2 , 2)).to(torch.uint8).npu()
        antiquant_scale = torch.randint(124, 130, (g, n, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()
        bias = torch.randint(-5, 5, (g, n)).to(output_dtype).npu()
        group_list = torch.Tensor([2, 4, 6, 16]).to(torch.int64).npu()
        group_list_type = 0
        model = NetPTA().npu()
        out = model(x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type, output_dtype)
        print("output")
        print(out[0].cpu())

    if __name__ == '__main__':
        main()
    ```
  <!-- end id77 -->
  <!-- npu="950" id78 -->
  - <term>Ascend 950PR&950DT products</term>: pergroup fake-quantization scenario example (x is `torch.float16`/`torch.bfloat16`, and weight uses `torch.int32` to carry `torch_npu.int4`, that is, one `torch.int32` value carries eight `torch_npu.int4` values)

    ```python
    import torch
    import torch_npu

    E, M, N, K, group_size = 2, 32, 128, 256, 32
    group_list = [M // E] * E
    xdtype = torch.bfloat16
    is_multi = True
    transpose_weight = True

    if __name__ == "__main__":
        assert not (is_multi and not transpose_weight), "multi/multi/multi and non-transpose is not supported"

        x = [torch.rand((group_list[i], K), dtype=xdtype, device="npu") for i in range(E)] if is_multi else \
            [torch.rand((M, K), dtype=xdtype, device="npu")]
        weight = [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (N, K) if transpose_weight else (K, N), dtype=torch.int32, device="npu"))
                    for _ in range(E)] if is_multi else \
                  [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (E, N, K) if transpose_weight else (E, K, N), dtype=torch.int32, device="npu"))]
        weight = [w.transpose(-1, -2) if transpose_weight else w for w in weight]
        bias = [torch.rand((N, ), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, N), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu")]
        antiquantScale = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]
        antiquantOffset = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]

        out = torch_npu.npu_grouped_matmul(x, weight, bias=bias,
                                            antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                            group_list=None if is_multi else torch.Tensor(group_list).to(torch.int64).npu(),
                                            split_item=0 if is_multi else 2,
                                            group_type=-1 if is_multi else 0,
                                            group_list_type=1)
        print([o.shape for o in out])
    ```

  <!-- end id78 -->
  <!-- npu="950" id79 -->
  - <term>Ascend 950PR&950DT products</term>: S8S4 K-C fake-quantization example

    ```python
    import os
    import numpy as np
    import torch
    import torch_npu

    def encode_a8w4_scale(scale_fp32: torch.Tensor) -> torch.Tensor:
        """Encode FP32 scales as the UINT64 carrier required by A8W4."""
        scale_np = scale_fp32.cpu().contiguous().numpy().astype(np.float32)
        scale_bits = scale_np.view(np.uint32)
        scale_bits &= np.uint32(0xFFFFE000)
        encoded = scale_bits.astype(np.uint64)
        encoded |= np.uint64(1 << 46)
        return torch.from_numpy(encoded.copy())

    def main() -> None:
        device_id = int(os.getenv("ASCEND_DEVICE_ID", "0"))
        torch.npu.set_device(device_id)

        m, n, k, expert_num = 64, 128, 256, 1

        x = torch.randint(-128, 128, (m, k), dtype=torch.int8).npu()

        # INT32 carries logical INT4 values in [-8, 7] before NPU packing.
        weight_i32 = torch.randint(
            -8, 8, (expert_num, k, n), dtype=torch.int32
        ).npu()
        weight_i4 = torch_npu.npu_convert_weight_to_int4pack(weight_i32)

        # A non-empty offset selects Per-channel mode. Both tensors are [E, 1, N].
        scale_fp32 = torch.rand(expert_num, 1, n) * 0.01 + 0.001
        scale = encode_a8w4_scale(scale_fp32).npu()
        offset = torch.zeros((expert_num, 1, n), dtype=torch.float32).npu()
        per_token_scale = (torch.rand(m, dtype=torch.float32) * 0.01 + 0.001).npu()
        bias = torch.zeros((expert_num, n), dtype=torch.float32).npu()
        group_list = torch.tensor([m], dtype=torch.int64).npu()

        output = torch_npu.npu_grouped_matmul(
            [x],
            [weight_i4],
            bias=[bias],
            scale=[scale],
            offset=[offset],
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[per_token_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
            act_type=0,
            tuning_config=[0],
            # Asymmetric S8S4 K-C (offset is non-empty) requires FP16 output.
            output_dtype=torch.float16,
        )[0]

        torch.npu.synchronize()
    ```
  <!-- end id79 -->
  <!-- npu="950" id80 -->
  - <term>Ascend 950PR&950DT products</term>: full quantization `torch_npu.int4`, perchannel example

    ```python
    import numpy as np
    import torch
    import torch_npu
    
    # The internal format (allow_internal_format) must be enabled when npu_quantize packs int4.
    torch_npu.npu.config.allow_internal_format = True
    
    
    def main(): 
        m, k, n, e = 2048, 1024, 1024, 16
        out_dtype = torch.bfloat16
        npu = "npu"
        print(f"[run] perchannel  shape M={m} K={k} N={n} E={e}  out={out_dtype}")
    
        x_int = torch.full((m, k), 1, dtype=torch.int8)
        w_int = torch.full((e, k, n), 1, dtype=torch.int8)
        scale_ref = torch.full((e, 1, n), 0.1, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        pt_scale = torch.full((m,), 1.0, dtype=torch.float32)
        group_list = torch.arange(1, e + 1) * (m // e)
    
        # ---- NPU side: int4 quantization packing (ND, non-transposed) ----
        x_quant = torch_npu.npu_quantize(
            x_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)
        weight_quant = torch_npu.npu_quantize(
            w_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)
    
        # ---- scale packing: the bits of (E,1,N) fp32 are placed into the lower 32 bits of int64, with the upper 32 bits = 0 ----
        scale_arr = scale_ref.cpu().numpy().astype(np.float32)
        scale_arr.dtype = np.uint32                               # bit reinterpretation, values unchanged
        packed = np.zeros((e, 1, n * 2), dtype=np.uint32)
        packed[..., ::2] = scale_arr
        packed.dtype = np.int64
        scale_i64 = torch.from_numpy(packed).to(npu)
    
        y = torch_npu.npu_grouped_matmul(
            [x_quant], [weight_quant],
            bias=None,
            scale=[scale_i64],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[pt_scale.to(npu)],
            group_list=group_list.to(npu),
            split_item=3,
            group_type=0,
            group_list_type=0,
            act_type=0,
            output_dtype=out_dtype,
        )[0]
        torch_npu.npu.synchronize()    
    
    if __name__ == "__main__":
        main()
    ```
  <!-- end id80 -->
  <!-- npu="950" id81 -->
  - <term>Ascend 950PR&950DT products</term>: full quantization `torch.int8`, dynamic K-C quantization with activation example

    ```python
    import torch
    import torch_npu
    x = torch.randint(-1, 1, (4, 4), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (2, 4, 5), dtype=torch.int8).npu()
    scale = torch.randint(-1, 1, (2, 5), dtype=torch.float32).npu()
    pertoken = torch.randint(-1, 1, (4,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (2, 5), dtype=torch.int32).npu()
    npu_out = torch_npu.npu_grouped_matmul([x], [weight1], scale=[scale], bias = [bias],    per_token_scale = [pertoken],group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, act_type=1)
    print("npu_out[0].shape: ", npu_out[0].shape)
    ```
  <!-- end id81 -->
  <!-- npu="950" id82 -->
  - <term>Ascend 950PR&950DT products</term>: fake-quantization MXA8W4, single-multi-single scenario

    ```python
    import torch
    import torch_npu
    MX_GROUP_SIZE = 32

    def test():
        # 1. Basic parameters
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        # 2. Generate Input x: (m, k)
        x_list = [torch.ones((m, k)).to(torch.float8_e4m3fn).npu()]
        # 3. Generate group_list
        m_per_group = m // g
        group_list = torch.tensor([m_per_group] * g, dtype=torch.int64).npu()
        group_list_type = 1  # the values in group_list indicate the size of each group on the grouping axis.
        # 4. Generate parameters
        weight_list = []
        antiquant_scale_list = []
        bias_list = []
        for i in range(g):
            # shape of a single group of weights: (n, k)
            w = torch.randint(0, 256, (n, k//2), dtype=torch.uint8).npu()
            w_cast = torch_npu.npu_format_cast(w, 29,
                                                customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2).t()
            weight_list.append(w_cast)
            # Assume that scale is also grouped or global. Generate the list here.
            s_k = k // MX_GROUP_SIZE // 2
            s = torch.randint(124, 130, (n, s_k, 2), dtype=torch.uint8).npu().permute(1, 0, 2)
            antiquant_scale_list.append(s)
            # Bias list (g values of (n))
            b = torch.randint(-5, 5, (n,), dtype=output_dtype).npu()
            bias_list.append(b)
        # 5. Prepare per_token_scale
        per_token_scale_list = [torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()]
        # 6. Execute
        out_list = torch_npu.npu_grouped_matmul(
            x_list,
            weight_list,
            bias=bias_list,
            group_list=group_list,
            group_type=0,
            group_list_type=group_list_type,
            per_token_scale=per_token_scale_list,
            antiquant_scale=antiquant_scale_list,
            split_item=2,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            weight_dtype=torch_npu.float4_e2m1fn_x2
        )
    if __name__ == '__main__':
        test()
    ```
  <!-- end id82 -->

- Graph mode call

  - Non-quantization scenario example:

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    M = 256
    N = 256
    K = 256
    g = 2
    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, split_item):
            return torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)

    def main():
        x1 = torch.ones(M, K, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, K, N, dtype=torch.float16)
        weight = [weight1.npu()]
        group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
        split_item = 3
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(x, weight, group_list, split_item)
        print(custom_output[0].cpu())

    if __name__ == '__main__':
        main()
    ```

  <!-- npu="950" id83 -->
  - <term>Ascend 950PR&950DT products</term>: full quantization scenario example

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2
      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]
          weight1 = torch.ones(g, K, N, dtype=torch.int8)
          weight = [weight1.npu()]
          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id83 -->
  <!-- npu="950" id84 -->
  - <term>Ascend 950PR&950DT products</term>: full quantization weightNz scenario example (x/weight are `torch.int8`)

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os
      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, [weight.transpose(1,2)], scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)
      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]

          weight3 = torch.ones(g, K, N, dtype=torch.int8).npu().transpose(1,2)
          weight = torch_npu.npu_format_cast(weight3, 29) # convert weight to the nz format

          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id84 -->
  <!-- npu="950" id85 -->
  - <term>Ascend 950PR&950DT products</term>: mx fake-quantization scenario example (x is `torch.float16`, weight is `torch.float32` (`torch_npu.float4_e2m1fn_x2`), and antiquant_scale uses `torch.uint8` to carry `torch.float8_e8m0fnu`)

      ```python
      import numpy as np
      from ml_dtypes import float4_e2m1fn
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      m = 1
      k = 64
      n = 64
      g = 2
      E2M1_MIN, E2M1_MAX = -6, 6
      antiquant_group_size = 32
      trans_weight = False

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, antiquant_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul(x, weight, antiquant_scale=antiquant_scale,
                                                  group_list=group_list,
                                                  split_item=split_item,
                                                  group_type=0, output_dtype=torch.float16, group_list_type=1)

      def main():
          x1 = torch.ones(m, k, dtype=torch.float16)
          x = [x1.npu()]
          weight1 = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(g * k * n).reshape((g, k, n))).astype(float4_e2m1fn)
          weight1 = torch.from_numpy(weight1.astype(np.float32))
          npu_weight = torch_npu.npu_format_cast(weight1.npu(), 29, customize_dtype=torch.float16)
          weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
          weight = [weight_packed.npu()]
          antiquant_scale1 = torch.randint(127, 128, (g, k // antiquant_group_size, n), dtype=torch.uint8)
          print(antiquant_scale1)
          antiquant_scale = [antiquant_scale1.npu()]
          group_list = torch.Tensor([0, 1]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=True)
          custom_output = model(x, weight, antiquant_scale, group_list, split_item)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id85 -->
  <!-- npu="950" id86 -->
  - <term>Ascend 950PR&950DT products</term>: G-B dynamic quantization scenario example

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 576
      N = 7168
      K = 512
      g = 4

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight1, scale1, group_list, split_item, pertoken):
              weight1 = weight1.transpose(1,2)
              scale1 = scale1.transpose(1,2)
              scale = [scale1]
              weight = [weight1]
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8).to(torch.float8_e4m3fn)
          x = [x1.npu()]
          weight1 = torch.ones(g, N, K, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
          scale1 = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken1 = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight1, scale1, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id86 -->
  <!-- npu="950" id87 -->
  - <term>Ascend 950PR&950DT products</term>: mx (A4W4) full quantization scenario example - mxfp4

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, per_token_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
        split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.float4_e2m1fn_x2, weight_dtype=torch_npu.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)

      def main():
          M = 64
          N = 128
          K = 128
          g = 1
          x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()
          weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()
          scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
          per_token_scale = torch.randint(-1, 1, (M,math.ceil(K/64),2), dtype=torch.int8).npu()
          group_list = torch.Tensor([1]).to(torch.int64).npu()
          split_item = 2
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, per_token_scale, group_list, split_item)

      if __name__ == '__main__':
          main()
      ```
  <!-- end id87 -->
