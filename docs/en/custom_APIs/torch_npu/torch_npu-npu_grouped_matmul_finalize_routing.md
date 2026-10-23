# torch\_npu.npu\_grouped\_matmul\_finalize\_routing

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

## Function<a name="en-us_topic_0000002259406069_section14441124184110"></a>

A fused operator that combines GroupedMatMul and MoeFinalizeRouting. GroupedMatMul performs separate computations for each expert, while MoeFinalizeRouting maps the GroupedMatMul results back to the original token positions and aggregates them based on the routing relationships. The two operations are performed in sequence to form a complete MoE output path.

- **MoE**: Mixture-of-Experts. Each token is assigned to one or more experts for computation based on the routing results.
- **MoeFinalizeRouting**: The finalization process of MoE routing. It maps the computation results of each expert back to the original token order based on routing indices and aggregates the results from multiple experts for the same token to produce the final output.

## Prototype<a name="en-us_topic_0000002259406069_section45077510411"></a>

```python
torch_npu.npu_grouped_matmul_finalize_routing(x, w, group_list, *, scale=None, bias=None, offset=None, pertoken_scale=None, shared_input=None, logit=None, row_index=None, dtype=None, shared_input_weight=1.0, shared_input_offset=0, output_bs=0, group_list_type=1, x_dtype=None, w_dtype=None, scale_dtype=None, pertoken_scale_dtype=None)
```

## Parameters<a name="en-us_topic_0000002259406069_section112637109429"></a>

- **`x`** (`Tensor`): Required. Left matrix for matrix computation. Non-contiguous tensors are not supported. The data type can be `torch.int8`. The data layout can be ND. The shape of this parameter is `(m, k)`. The value range of `m` is [1, 16 \* 1024 \* 8].

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.int8`.
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, the data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.float4_e2m1fn_x2`. For the float4 series, the optional parameter `x_dtype` must be set to the corresponding type; in this case, the `dtype` of `x` itself no longer takes effect, but the `dtype` of `x` itself must still be an 8-bit data type to ensure the shape is correct. For the float4 series, the inner axis `K` must be an even number to ensure that 8 bits can be converted into two float4 values.
    - In pertoken quantization scenarios, the data type can be `torch.int8`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`.
    - `x` supports m of 0. Other empty tensor scenarios are not supported.
  <!-- end id5 -->

- **`w`** (`Tensor`): Required. Right matrix for matrix computation. Non-contiguous tensors are not supported. The data type can be `torch.int8` or `torch_npu.int4`.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.int8` or `torch_npu.int4`.
    - In A8W8 quantization scenarios, the data layout can be `FRACTAL_NZ`, and the shape of this parameter is `(e, n1, k1, k0, n0)`, where `k0` is fixed at `16` and `n0` is fixed at `32`. The `k` dimension in the shape of `x` and the `k1` dimension in the shape of `w` must satisfy the equation: $\mathrm{ceilDiv}(k, 16) = k1$. The value of `e` must be in the range [1, 256]. The value of `k` must be a multiple of `16`. The value of `n` must be a multiple of `32` and greater than or equal to `256`.
    - In A8W4 quantization scenarios, the data layout can be ND, and the shape of this parameter is `(e, k, n)`, where `k` is fixed at `2048` and `n` is fixed at `7168`.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, the data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, or `torch_npu.float4_e2m1fn_x2`. For the float4 series, the optional parameter `w_dtype` must be set to the corresponding type; in this case, the `dtype` of `w` itself no longer takes effect, but the `dtype` of `w` itself must still be an 8-bit data type to ensure the shape is correct. In the float4 scenario, the `K` of the input `x` must be an even number greater than 2, and when `weight` is not transposed, the inner axis `N` must be an even number to ensure that 8 bits can be converted into two float4 values. The dimensions are `(e, k, n)`.
    - In pertoken quantization scenarios, the data type can be `torch.int8`, `torch.float8_e4m3fn`, or `torch_npu.hifloat8`. The data layout supports `ND` and `FRACTAL_NZ`, and the `torch_npu.npu_format_cast` API can be used to convert ND to the `FRACTAL_NZ` format.
    - `w` supports n of 0. Other empty tensor scenarios are not supported.
  <!-- end id7 -->

- **`*`**: Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`group_list`** (`Tensor`): Required. Group sizes for `GroupedMatMul`. Non-contiguous tensors are supported. The data type can be `torch.int64`. The data layout can be ND. The shape of this parameter is `(e,)`, where `e` is identical to that of `w`. The sum of all elements in `group_list` must be less than or equal to `m`. The length of `group_list` must not exceed 1024.
- **`scale`** (`Tensor`): Optional. Dequantization parameter for the matrix computation, corresponding to the `weight` matrix. The data layout supports ND.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: In A8W8 scenarios, per-channel quantization is supported. Non-contiguous tensors are not supported. The data type can be `torch.float32`, and the dimensions are `(e, n)`, where n = n1*n0. In A8W4 quantization scenarios, the data type can be `torch.int64`, and the dimensions are `(e, 1, n)`.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, the data type can be `torch.float8_e8m0fnu`. When this format is used, `scale_dtype` must be set to `torch_npu.float8_e8m0fnu`; in this case, the `dtype` of `scale` itself no longer takes effect, but the `dtype` of `scale` itself must still be an 8-bit data type to ensure the shape is correct. The dimensions are `(e, k/64, n, 2)` when `weight` is not transposed, and `(e, n, k/64, 2)` when `weight` is transposed.
    - In pertoken quantization scenarios, the data type can be `torch.float32` or `torch.bfloat16`. The dimensions are `(e, 1, n)`, where `e` and `n` are the same as the `e` and `n` of `w`.
  <!-- end id9 -->

- **`bias`** (`Tensor`): Optional. Bias parameter for matrix computation. Non-contiguous tensors are not supported. The data type can be `torch.float32`. The data layout can be ND. The shape of this parameter is `(e, n)`. This parameter is supported only in A8W4 quantization scenarios.

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the A8W4 scenario is supported, and the data type can be `torch.float32`.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The mx quantization and pertoken quantization scenarios are supported, and the data type can be `torch.bfloat16`.
  <!-- end id11 -->

- **`offset`** (`Tensor`): Optional. Offset for matrix quantization parameters. Non-contiguous tensors are not supported. The data type can be `torch.float32`. The data layout can be ND. This parameter must be 3D. This parameter is supported only in A8W4 quantization scenarios.

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the A8W4 quantization scenario is supported.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter is not supported currently. Use the default value.
  <!-- end id13 -->

- **`pertoken_scale`** (`Tensor`): Optional. Dequantization parameter for the `x` matrix under per-token quantization. Non-contiguous tensors are not supported. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`. The data type can be `torch.float32`. The data layout can be ND.

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float32`. The dimensions are `(m,)`.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, this is a **required** parameter, and the data type can be `torch.float8_e8m0fnu`. When this format is used, `pertoken_scale_dtype` must be set to `torch_npu.float8_e8m0fnu`; in this case, the `dtype` of `pertoken_scale` itself no longer takes effect, but the `dtype` of `pertoken_scale` itself must still be an 8-bit data type to ensure the shape is correct. The dimensions are `(m, k/64, 2)`.
    - In pertoken quantization scenarios, this is an **optional** parameter, and the data type can be `torch.float32`. The dimensions are `(m,)`.
  <!-- end id15 -->

- **`shared_input`** (`Tensor`): Optional. Output of the shared expert in MoE computation, which must be combined with the MoE expert output. Non-contiguous tensors are not supported. The data type can be `torch.bfloat16`. The data layout can be ND. This shape of this parameter is `(batch/dp, n)`, where `n` is identical to that of `scale`. The value range of `batch/dp` is [1, 2 \* 1024]. The value range of `batch` is [1, 16 \* 1024].
- **`logit`** (`Tensor`): Optional. Per-token logit values from MoE experts. The output of matrix multiplication is multiplied by these logit values and then combined based on indices. Non-contiguous tensors are not supported. The data type can be `torch.float32`. The data layout can be ND. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`.

  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: This parameter must be passed.
  <!-- end id16 -->

- **`row_index`** (`Tensor`): Optional. Output of MoE experts is combined based on this row index, where the values serve as indices for scatter-add combination. Non-contiguous tensors are not supported. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. The shape of this parameter is `(m,)`, where `m` is identical to that of `x`.

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.int32` or `torch.int64`.
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, this is a required parameter, and the data type can be `torch.int64`.
    - In pertoken quantization scenarios, this is a **required** parameter. When the data type of the input `x` is `torch.int8`, the data type of `row_index` can be `torch.int64` or `torch.int32`. When the data type of the input `x` is `torch.float8_e4m3fn` or `torch_npu.hifloat8`, the data type of `row_index` can be `torch.int64`;
  <!-- end id18 -->

- **`dtype`** (`ScalarType`): Optional. Output type of the `GroupedMatMul` computation. `0` indicates `torch.float32`, `1` indicates `torch.float16`, and `2` indicates `torch.bfloat16`. The default value is 0.
- **`shared_input_weight`** (`float`): Optional. Factor for combining shared expert output with MoE expert output. The `shared_input` is multiplied by this parameter before accumulating with the MoE expert results. The default value is `1.0`.
- **`shared_input_offset`** (`int`): Optional. Row offset for combining shared expert and MoE expert outputs. The default value is `0`, indicating no offset.
- **`output_bs`** (`int`): Optional. Maximum size of the output batch dimension. The default value is `0`.
- **`group_list_type`** (`int`): Optional. Grouping mode for `GroupedMatMul`. Valid values are: `1` (count mode) or `0` (cumsum mode, representing the prefix sum). The default value is `1`.
- **`x_dtype`** (`int`): Optional. Real data type of the input `x`. The default value is None.

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not supported currently. Use the default value.
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, if None is passed, the real data type of the input `x` is the same as the input `dtype`. If a value other than None is passed, the real data type of `x` can be `torch_npu.float4_e2m1fn_x2`.
    - In pertoken quantization scenarios, if None is passed, the real data type of the input `x` is the same as the input `dtype`. If a value other than None is passed, the real data type of `x` can be `torch_npu.hifloat8`.
  <!-- end id20 -->

- **`w_dtype`** (`int`): Optional. Real data type of the input `w`. The default value is None.

  <!-- npu="A3,910b" id21 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not supported currently. Use the default value.
  <!-- end id21 -->
  <!-- npu="950" id22 -->
  - <term>Ascend 950PR&950DT products</term>:
    - In mx quantization scenarios, if None is passed, the real data type of the input `x` is the same as the input `dtype`. If a value other than None is passed, the real data type of `x` can be `torch_npu.float4_e2m1fn_x2`.
    - In pertoken quantization scenarios, if None is passed, the real data type of the input `x` is the same as the input `dtype`. If a value other than None is passed, the real data type of `x` can be `torch_npu.hifloat8`.
  <!-- end id22 -->

- **`scale_dtype`** (`int`): Optional. Real data type of the input `scale`. The default value is None.

  <!-- npu="A3,910b" id23 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not supported currently. Use the default value.
  <!-- end id23 -->
  <!-- npu="950" id24 -->
  - <term>Ascend 950PR&950DT products</term>: In mx quantization scenarios, if None is passed, the real data type of the input `scale` is the same as the input `dtype`. If a value other than None is passed, this parameter must be set to `torch_npu.float8_e8m0fnu`, indicating that `scale` uses the `torch.float8_e8m0fnu` format.
  <!-- end id24 -->

- **`pertoken_scale_dtype`** (`int`): Optional. Real data type of the input `pertoken_scale`. The default value is None.

  <!-- npu="A3,910b" id25 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: This parameter is not supported currently. Use the default value.
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT products</term>: In mx quantization scenarios, if None is passed, the real data type of the input `pertoken_scale` is the same as the input `dtype`. If a value other than None is passed, this parameter must be set to `torch_npu.float8_e8m0fnu`, indicating that `pertoken_scale` uses the `torch.float8_e8m0fnu` format.
  <!-- end id26 -->

## Return Values<a name="en-us_topic_0000002259406069_section22231435517"></a>

**`y`** (`Tensor`): A 2D tensor. Non-contiguous tensors are supported. The output data type is fixed at `torch.float32`, and the shape is `(batch, n)`.

## Constraints<a name="en-us_topic_0000002259406069_section12345537164214"></a>

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- The following table lists the data type combinations supported by the input and output tensors.

  <!-- npu="A3,910b" id27 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    | x | w | group_list | scale | bias | offset | pertoken_scale | shared_input | logit | row_index | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | None | None | `torch.float32` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | None | None | `torch.float32` | None | None | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch_npu.int4` | `torch.int64` | `torch.int64` | `torch.float32` | None | `torch.float32` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch_npu.int4` | `torch.int64` | `torch.int64` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
  <!-- end id27 -->
  <!-- npu="950" id28 -->
  - <term>Ascend 950PR&950DT products</term>:

    | x | w | group_list | scale | bias | offset | pertoken_scale | shared_input | logit | row_index | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float8_e5m2` | `torch.float8_e5m2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.float8_e5m2` | `torch.float8_e4m3fn` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.float8_e4m3fn` | `torch.float8_e5m2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.bfloat16` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int32` | `torch.float32` |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.bfloat16` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int32` | `torch.float32` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch.float32` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch.bfloat16` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.float32` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.bfloat16` | `torch.bfloat16`/None | None | `torch.float32`/None | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |
    | `torch.float8_e4m3fn` | `torch_npu.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.bfloat16`/None | None | `torch.float8_e8m0fnu` | `torch.bfloat16` | `torch.float32` | `torch.int64` | `torch.float32` |

    > **MxA8W4 scenario description**:
    > - The data type of `x` is `torch.float8_e4m3fn`, and the data type of `w` is `torch_npu.float4_e2m1fn_x2`. The data layout of `w` must be the FRACTAL_NZ format, and the torch_npu.npu_format_cast API can be used to convert ND to the FRACTAL_NZ format.
    > - `x` must not be transposed, and `w` must be transposed. The shape of `w` is `(e, n, k)`, where e <= 1024, and k and n must be aligned to 32. In graph mode scenarios, e must not be 1 and k > 64.
    > - The sum of the first dimensions of `shared_input_offset` and `shared_input` must not exceed `output_bs`.
  <!-- end id28 -->

## Examples<a name="en-us_topic_0000002259406069_section14459801435"></a>

- Single-operator call

  <!-- npu="A3,910b" id29 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    ```python
    import numpy as np
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8

    x = np.random.randint(-1, 1, (m, k)).astype(np.int8)
    weight = np.random.randint(-1, 1, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch
    y = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNz,
                group_list_clone, scale=scale_clone, pertoken_scale=pertoken_scale_clone,
                shared_input=shared_input_clone, logit=logit_clone, row_index=row_index_clone,
                shared_input_offset=shared_input_offset, output_bs=output_bs)
    ```
  <!-- end id29 -->
  <!-- npu="950" id30 -->
  - <term>Ascend 950PR&950DT products</term>: mx quantization scenario example - mxfp8

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    from scipy.special import softmax
    def gen_input_data_mxfp8_wtrans(M, K, N, E):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
        Scale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        pertoken_scale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64).npu()
        return x, weight, Scale, pertoken_scale, groupList
    m, k, n, batch, topK, group_num = 72, 32, 7168, 72, 1, 2
    x_clone, weight_clone, scale_clone, pertoken_scale_clone, group_list_clone = gen_input_data_mxfp8_wtrans(m, k, n, group_num)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = torch_npu.float8_e8m0fnu
    pertoken_scale_dtype = torch_npu.float8_e8m0fnu
    # w transpose
    weight_clone = weight_clone.transpose(1,2)
    scale_clone = scale_clone.transpose(1,2)
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    out = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weight_clone, group_list_clone,bias=bias,
        scale=scale_clone, pertoken_scale=pertoken_scale_clone, shared_input=shared_input_clone,
        logit=logit_clone, row_index=row_index_clone, shared_input_offset=shared_input_offset, output_bs=output_bs,
        group_list_type=group_list_type,x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
    print(out.cpu())
    print(out.cpu().shape)
    ```
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT products</term>: pertoken quantization weightNZ scenario example - `torch.int8`

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    from scipy.special import softmax
    m, k, n, batch, topK, group_num = 72, 32, 7168, 72, 1, 2
    x_clone = torch.randint(-128, 127, (m, k), dtype=torch.int8).npu()
    weight_clone = torch.randint(-128, 127, (group_num, k, n), dtype=torch.int8).npu()
    # w transpose
    # weight_clone = weight_clone.transpose(1,2)
    weightNZ_clone = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.randint(low=0, high=256, size=(group_num, 1, n), dtype=torch.float32).npu()
    pertoken_scale_clone = (torch.rand(m, dtype=torch.float32) * 256).npu()
    group_list_clone = torch.tensor([m//2, m//2], dtype=torch.int64).npu()
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = None
    pertoken_scale_dtype = None
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    out = torch_npu.npu_grouped_matmul_finalize_routing(x_clone, weightNZ_clone, group_list_clone,bias=bias,
        scale=scale_clone, pertoken_scale=pertoken_scale_clone, shared_input=shared_input_clone,
        logit=logit_clone, row_index=row_index_clone, shared_input_offset=shared_input_offset, output_bs=output_bs,
        group_list_type=group_list_type,x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
    print(out.cpu())
    print(out.cpu().shape)
    ```
  <!-- end id31 -->
  <!-- npu="950" id32 -->
  - <term>Ascend 950PR&950DT products</term>: fake-quantization scenario mxA8W4

    ```python
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    import torch
    import torch_npu
    import math

    def ceil_div(a, b):
        return math.ceil(a / b)
    def fp32_to_fp4_e2m1_u8packed(tensor_in):
        fp4_values = np.array([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=np.float32)
        x = tensor_in.numpy()
        x_flat = x.reshape(-1, 1)  # (N, 1)
        fp4_values = fp4_values.reshape(1, -1)  # (1, M)
        dist = np.abs(x_flat - fp4_values)  # (N, M)
        indices = np.argmin(dist, axis=1)  # (N,)
        tmp = indices.reshape(-1, 2)
        packed = tmp[:, 0] + (tmp[:, 1] << 4)
        packed = packed.astype(np.uint8)
        shape_out = list(tensor_in.shape)
        shape_out[-1] = shape_out[-1] // 2
        out = torch.from_numpy(packed).reshape(shape_out)
        return out

    g, group_size, m, k, n = 4, 32, 128, 64, 512
    group_list_type = 1  # 0: cumsun 1: count
    output_dtype = torch.float32
    shared_input_weight = 0.5
    shared_input_offset = 2
    output_bs = 8
    # generate data
    x_range = [-1, 1]
    weight_range = [-6, 6]
    scale_range = [0, 2]
    bias_range = [0, 2]
    pertoken_scale_range = [0, 2]
    logit_range = [0, 2]
    shared_input_range = [0, 2]
    row_index_range = [0, 2]
    x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
    x = x.to(torch.float8_e4m3fn)
    weight = torch.rand((g, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
    weight = fp32_to_fp4_e2m1_u8packed(weight)
    pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        pertoken_scale_range[1] - pertoken_scale_range[0]) + pertoken_scale_range[0]
    pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    scale = torch.rand((g, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    bias = torch.rand((g, n), dtype=torch.bfloat16) * (
        bias_range[1] - bias_range[0]) + bias_range[0]
    logit = torch.rand((m,), dtype=torch.float32) * (logit_range[1] - logit_range[0]) + logit_range[0]
    shared_input = torch.rand((output_bs // 2, n), dtype=torch.bfloat16) * (
        shared_input_range[1] - shared_input_range[0]) + shared_input_range[0]
    row_index = torch.randint(low=row_index_range[0], high=row_index_range[1], size=(m,), dtype=torch.int64)
    group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
    weight_npu = weight.npu()
    weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2)
    # npu
    x_npu = x.npu()
    pertoken_scale_npu = pertoken_scale.npu()
    scale_npu = scale.npu()
    bias_npu = bias.npu()
    logit_npu = logit.npu()
    shared_input_npu = shared_input.npu()
    row_index_npu = row_index.npu()
    group_list_npu = group_list.npu()
    weight_npu = weight_npu.transpose(-1, -2)
    scale_npu = scale_npu.transpose(-2, -3)
    output = torch_npu.npu_grouped_matmul_finalize_routing(x_npu, weight_npu, group_list=group_list_npu,             scale=scale_npu,bias=bias_npu, pertoken_scale=pertoken_scale_npu, shared_input=shared_input_npu,               logit=logit_npu,row_index=row_index_npu,shared_input_weight=shared_input_weight,               shared_input_offset=shared_input_offset, group_list_type=group_list_type,               output_bs=output_bs, scale_dtype=torch_npu.float8_e8m0fnu, pertoken_scale_dtype=torch_npu.float8_e8m0fnu,               w_dtype=torch_npu.float4_e2m1fn_x2)
    ```
  <!-- end id32 -->

- Graph mode call

  <!-- npu="A3,910b" id33 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:

    ```python
    import numpy as np
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,
                        scale=scale, pertoken_scale=pertoken_scale, shared_input=shared_input,
                        logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs)
            return output

    m, k, n = 576, 2048, 7168
    batch = 72
    topK = 8
    group_num = 8

    x = np.random.randint(-10, 10, (m, k)).astype(np.int8)
    weight = np.random.randint(-10, 10, (group_num, k, n)).astype(np.int8)
    scale = np.random.normal(0, 0.01, (group_num, n)).astype(np.float32)
    pertoken_scale = np.random.normal(0, 0.01, (m, )).astype(np.float32)
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)

    x_clone = torch.from_numpy(x).npu()
    weight_clone = torch.from_numpy(weight).npu()
    weightNz = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.from_numpy(scale).npu()
    pertoken_scale_clone = torch.from_numpy(pertoken_scale).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    output_bs = batch

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNz, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs)
    ```
  <!-- end id33 -->
  <!-- npu="950" id34 -->
  - <term>Ascend 950PR&950DT products</term>: mx quantization scenario example - mxfp4 graph mode call

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    class Model(torch.nn.Module):
        import torch
        import torch_npu
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, bias, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs, group_list_type,
            x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype):
            weight = weight.transpose(1, 2)
            scale = scale.transpose(1, 2)
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,bias=bias,
            scale=scale, pertoken_scale=pertoken_scale, shared_input=None,
            logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs, group_list_type=group_list_type,
                x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
            return output
    m, k, n = 72, 32, 7168
    batch = 72
    topK = group_num = 1
    scale_clone = torch.full((group_num, n, k//64, 2), 1, dtype=torch.int8).npu()
    pertoken_scale_clone = torch.full((m, k//64, 2), 1, dtype=torch.int8).npu()
    group_list = np.array([batch] * group_num, dtype=np.int64)
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    x_clone = torch.randint(1, 2, (m, k // 2), dtype=torch.uint8).npu()
    weight_clone = torch.randint(1, 2, (group_num, n // 2, k), dtype=torch.uint8).npu()
    group_list_clone = torch.from_numpy(group_list).npu()
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype=torch_npu.float4_e2m1fn_x2
    w_dtype=torch_npu.float4_e2m1fn_x2
    scale_dtype=torch_npu.float8_e8m0fnu
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weight_clone, bias, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs, group_list_type, x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype)
    print(y.cpu())
    print(y.cpu().shape)
    ```
  <!-- end id34 -->
  <!-- npu="950" id35 -->
  - <term>Ascend 950PR&950DT products</term>: pertoken quantization weightNZ scenario example - `torch.int8` graph mode call

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    class Model(torch.nn.Module):
        import torch
        import torch_npu
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, bias, group_list, scale, pertoken_scale, shared_input, logit, row_index, shared_input_offset, output_bs, group_list_type,
            x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype):
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list,bias=bias,
            scale=scale, pertoken_scale=pertoken_scale, shared_input=None,
            logit=logit, row_index=row_index, shared_input_offset=shared_input_offset, output_bs=output_bs, group_list_type=group_list_type,
                x_dtype=x_dtype, w_dtype=w_dtype, scale_dtype=scale_dtype, pertoken_scale_dtype=pertoken_scale_dtype)
            return output
    m, k, n, batch, topK, group_num = 72, 256, 7168, 72, 1, 2
    x_clone = torch.randint(-128, 127, (m, k), dtype=torch.int8).npu()
    weight_clone = torch.randint(-128, 127, (group_num, k, n), dtype=torch.int8).npu()
    # w transpose
    # weight_clone = weight_clone.transpose(1,2)
    weightNZ_clone = torch_npu.npu_format_cast(weight_clone, 29)
    scale_clone = torch.randint(low=0, high=256, size=(group_num, 1, n), dtype=torch.float32).npu()
    pertoken_scale_clone = (torch.rand(m, dtype=torch.float32) * 256).npu()
    group_list_clone = torch.tensor([m//2, m//2], dtype=torch.int64).npu()
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input_clone = torch.from_numpy(shared_input).to(torch.bfloat16).npu()
    logit_ori = np.random.normal(0, 0.1, (batch, group_num)).astype(np.float32)
    routing = np.argsort(logit_ori, axis=1)[:, -topK:].astype(np.int32)
    logit = softmax(logit_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1).astype(np.float32)
    logit = logit.reshape(m)
    row_index = (np.argsort(routing.reshape(-1)) // topK).astype(np.int64)
    logit_clone = torch.from_numpy(logit).npu()
    row_index_clone = torch.from_numpy(row_index).npu()
    shared_input_offset = batch // 2
    group_list_type = 1
    output_bs = batch
    x_dtype = None
    w_dtype = None
    scale_dtype = None
    pertoken_scale_dtype = None
    bias = torch.randint(-1, 1, (group_num, n), dtype=torch.bfloat16).npu()
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x_clone, weightNZ_clone, bias, group_list_clone, scale_clone, pertoken_scale_clone, shared_input_clone, logit_clone, row_index_clone, shared_input_offset, output_bs, group_list_type, x_dtype, w_dtype, scale_dtype, pertoken_scale_dtype)
    print(y.cpu())
    print(y.cpu().shape)
    ```
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT products</term>: fake-quantization mxA8W4 scenario graph mode

    ```python
    import numpy as np
    from ml_dtypes import float4_e2m1fn
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    import math
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    # config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, pertoken_scale, scale, bias, logit, shared_input, row_index, group_list,
                    group_list_type,
                    output_bs, shared_input_weight, shared_input_offset):
            weight = weight.transpose(-1, -2)
            scale = scale.transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_finalize_routing(x, weight, group_list=group_list,
                                                                    scale=scale,
                                                                    bias=bias,
                                                                    pertoken_scale=pertoken_scale,
                                                                    shared_input=shared_input,
                                                                    logit=logit,
                                                                    row_index=row_index,
                                                                    shared_input_weight=shared_input_weight,
                                                                    shared_input_offset=shared_input_offset,
                                                                    group_list_type=group_list_type,
                                                                    output_bs=output_bs,
                                                                    scale_dtype=torch_npu.float8_e8m0fnu,
                                                                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    w_dtype=torch_npu.float4_e2m1fn_x2)
            return output

    def ceil_div(a, b):
        return math.ceil(a / b)

    def fp32_to_fp4_e2m1_u8packed(tensor_in):
        fp4_values = np.array([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=np.float32)
        x = tensor_in.numpy()
        x_flat = x.reshape(-1, 1)  # (N, 1)
        fp4_values = fp4_values.reshape(1, -1)  # (1, M)
        dist = np.abs(x_flat - fp4_values)  # (N, M)
        indices = np.argmin(dist, axis=1)  # (N,)
        tmp = indices.reshape(-1, 2)
        packed = tmp[:, 0] + (tmp[:, 1] << 4)
        packed = packed.astype(np.uint8)
        shape_out = list(tensor_in.shape)
        shape_out[-1] = shape_out[-1] // 2
        out = torch.from_numpy(packed).reshape(shape_out)
        return out

    g, group_size, m, k, n, is_dynamic = 4, 32, 128, 256, 512, True
    # mode controls single-operator/graph mode, static graph: 1; dynamic graph: 2
    mode = 2
    group_list_type = 1  # 0: cumsun 1: count
    output_dtype = torch.float32
    shared_input_weight = 0.5
    shared_input_offset = 2
    output_bs = 8
    # generate data
    x_range = [-1, 1]
    weight_range = [-6, 6]
    scale_range = [0, 2]
    bias_range = [0, 2]
    pertoken_scale_range = [0, 2]
    logit_range = [0, 2]
    shared_input_range = [0, 2]
    row_index_range = [0, 2]
    x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
    x = x.to(torch.float8_e4m3fn)
    weight = torch.rand((g, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
    weight = fp32_to_fp4_e2m1_u8packed(weight)
    pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        pertoken_scale_range[1] - pertoken_scale_range[0]) + pertoken_scale_range[0]
    pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    scale = torch.rand((g, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
        scale_range[1] - scale_range[0]) + scale_range[0]
    scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    bias = torch.rand((g, n), dtype=torch.bfloat16) * (
        bias_range[1] - bias_range[0]) + bias_range[0]
    logit = torch.rand((m,), dtype=torch.float32) * (logit_range[1] - logit_range[0]) + logit_range[0]
    shared_input = torch.rand((output_bs // 2, n), dtype=torch.bfloat16) * (
        shared_input_range[1] - shared_input_range[0]) + shared_input_range[0]
    row_index = torch.randint(low=row_index_range[0], high=row_index_range[1], size=(m,), dtype=torch.int64)
    group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
    x_npu = x.npu()
    weight_npu = weight.npu()
    pertoken_scale_npu = pertoken_scale.npu()
    scale_npu = scale.npu()
    bias_npu = bias.npu()
    logit_npu = logit.npu()
    shared_input_npu = shared_input.npu()
    row_index_npu = row_index.npu()
    group_list_npu = group_list.npu()
    weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2)
    model = NetPTA().npu()
    if mode == 1:
        model = torch.compile(model, backend=npu_backend, dynamic=False)
    elif mode == 2:
        model = torch.compile(model, backend=npu_backend, dynamic=True)
    out = model(x_npu, weight_npu, pertoken_scale_npu, scale_npu, bias_npu, logit_npu, shared_input_npu, row_index_npu,group_list_npu, group_list_type, output_bs, shared_input_weight, shared_input_offset)
    ```
  <!-- end id36 -->
