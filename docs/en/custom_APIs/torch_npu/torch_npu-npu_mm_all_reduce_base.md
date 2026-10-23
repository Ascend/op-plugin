# torch_npu.npu_mm_all_reduce_base

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- Fuses matrix multiplication (MatMul) and all_reduce collective communication operations in tensor parallelism (TP) scenarios. The fused kernel implements pipelined parallelism between computation and communication.

- Formula:
    $$
    output = allreduce\left(x1 \mathbin{@} \left((x2 + antiquantOffset) * antiquantScale\right) + bias + x3\right)
    $$

> [!NOTE]
> When using this API, ensure that the driver firmware package and CANN package are both version 8.0.RC2 or a matching later version. Otherwise, errors such as a BUS ERROR will be raised.

## Prototype

```python
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0, group_sizes=None, y_dtype=None, x1_dtype=None, x2_dtype=None, dequant_scale_dtype=None, pertoken_scale_dtype=None, comm_quant_mode=0, comm_mode=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. The data type can be `torch.int8`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. This parameter must be a 2D or 3D tensor.

    <!-- npu="910b" id3 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8`, `torch.float16`, or `torch.bfloat16`.
    <!-- end id3 -->
    <!-- npu="950" id4 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.hifloat8`, or `torch_npu.float4_e2m1fn_x2`.
    <!-- end id4 -->

- **`x2`** (`Tensor`): Required. The data type can be `torch.float16`, `torch.int8`, or `torch.bfloat16`. The data layout can be NZ (Ascend-optimized layout) or ND. In non-quantization scenarios, the data type must be identical to that of `x1`. The size of the 0th dimension must be identical to that of the last dimension of `x1`.

    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data layout supports NZ (Ascend-friendly layout) and ND.
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - <term>Ascend 950PR&950DT products</term>: The data layout supports ND.
    <!-- end id6 -->

- **`hcom`** (`str`): Required. Communicator handle name obtained by calling the `get_hccl_comm_name` API.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`reduce_op`** (`str`): Optional. Type of the reduce operation. Currently, only the default value `sum` is supported.
- **`bias`** (`Tensor`): Optional. The data type can be `torch.int32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. This parameter must be a 1D tensor, where the size must match the last dimension of `output` or `x2`.

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int32`, `torch.float16`, or `torch.bfloat16`.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int32`, `torch.float16`, `torch.bfloat16`, or `torch.float32`. In perblock scenarios, only an empty `bias` input is supported.
    <!-- end id8 -->

- **`antiquant_scale`** (`Tensor`): Optional. Dequantization scale for `x2` in fake-quantization scenarios, $antiquantScale$ in the formula. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. In fake-quantization scenarios, the data type must be identical to that of `x1`.
    - `pertensor` scenario: The shape is `[1]`.
    - `perchannel` scenario: The shape is `[1, n]` or `[n]`, where `n` indicates the size of the last dimension of `x2`.
    - `pergroup` scenario: The shape is `[ceil(k, antiquant_group_size), n]`, where `k` indicates the size of the first dimension of `x2`, `n` indicates the size of the last dimension of `x2`, and `antiquant_group_size` is the group size input for the dequantization computation of the input `x2`.

        > [!NOTE]
        > The calculation logic for $ceil(k, antiquant\_group\_size)$ is $(k + antiquant\_group\_size - 1)/antiquant\_group\_size$, where only the integer part of the result is taken.

- **`antiquant_offset`** (`Tensor`): Optional. Dequantization offset for `x2` in fake-quantization scenarios, $antiquantOffset$ in the formula. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. The data type and shape must be identical to those of `antiquant_scale`.
- **`x3`** (`Tensor`): Optional. Offset added after MatMul computation. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. The data type and shape must be identical to those of `output`.

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
    <!-- end id10 -->

- **`dequant_scale`** (`Tensor`): Optional. The default value is `None`. Dequantization scale applied after the MatMul computation. The data layout supports ND. The following quantization scenarios are supported:
    - `pertensor` scenario: The shape is `[1]`.
    - `perchannel` scenario: The shape is `[n]` or `[1, n]`, where `n` indicates the size of the last dimension of `x2`.
    - mx quantization scenario: When the data type is `torch.float8_e8m0fnu`, only transposed input is supported. When the shape of `x2` is `[n, k]`, the shape of x2Scale is `[n, ceilDiv(k, 64), 2]`, and ceilDiv(k, 32) must be an even number.
    - perblock scenario: The shape is `[ceilDiv(n, 128), ceilDiv(k, 128)]`.

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: The `pertensor` and `perchannel` scenarios are supported. The data type can be `torch.int64`, `torch.uint64`, `torch.bfloat16`, or `torch.float32`.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: The `pertensor`, `perchannel`, `mx`, and `perblock` scenarios are supported. The data type can be `torch.int64`, `torch.uint64`, `torch.bfloat16`, `torch.float32`, or `torch.float8_e8m0fnu`.
    <!-- end id12 -->

- **`pertoken_scale`** (`Tensor`): Optional. Per-token dequantization scale applied after MatMul computation. The data type can be `torch.float32`. When `x1` is `[m, k]`, the shape is `[m]`. When `x1` is `[b, s, k]`, the shape is `[b * s]`.

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32`.
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32` or `torch.float8_e8m0fnu`. If the data type is `torch.float8_e8m0fnu`: in mx scenarios, the shape is `[s, ceilDiv(k, 64), 2]` or `[m, ceilDiv(k, 64), 2]`, and ceilDiv(k, 32) must be an even number; in perblock scenarios, the shape is `[ceilDiv(m, 128), ceilDiv(k, 128)]`.
    <!-- end id14 -->

- **`comm_quant_scale_1`** (`Tensor`): Optional. Quantization and dequantization scale before and after AlltoAll communication. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. When `x2` is `[k, n]`, the shape is `[1, n]` or `[n]`. Ensure that the data on each rank is consistent and correct.
- **`comm_quant_scale_2`** (`Tensor`): Optional. Quantization and dequantization scale before and after AllGather communication. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND. When `x2` is `[k, n]`, the shape is `[1, n]` or `[n]`. Ensure that the data on each rank is consistent and correct.
- **`comm_turn`** (`int`): Optional. Communication splitting granularity between ranks. The default value is `0`, indicating the default splitting mode. Currently, only the value `0` is supported.
- **`antiquant_group_size`** (`int`): Optional. Group size for dequantizing `x2` in `pergroup` fake-quantization. It describes the size of the data block to be dequantized along the $k$ axis corresponding to a single set of dequantization parameters. When the fake-quantization mode is not `pergroup`, the value of this parameter must be `0`. When the fake-quantization mode is `pergroup`, the value of this parameter must be a multiple of 32 and must be in the range [32, min(k-1, INT_MAX)], where `k` indicates the size of the first dimension of `x2`. The default value is `0`, indicating a non-`pergroup` scenario.
- **`group_sizes`** (`List[int]`): Optional. The default value is `None`. Indicates how many values of the `x1`/`x2` input in the corresponding dimension direction can be dequantized by one x1Scale/x2Scale input value. group_sizes is [groupSizeM, groupSizeN, groupSizeK]. groupSizeM, groupSizeN, and groupSizeK indicate the number of values corresponding to one dequantization coefficient in each dimension. Automatic parameter inference is supported. When any one or more of groupSizeM/groupSizeN/groupSizeK derived from the formula is 0, the operator automatically infers the corresponding parameter value. The inference principle is as follows: assuming groupSizeM=0, the quantization group value in the m direction is inferred by the API, and the inference formula is groupSizeM = m / scaleM (m must be divisible by scaleM), where m is the same as the m in the `x1` shape, and scaleM is the same as the m in the `x1Scale` shape.

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id15 -->

- **`y_dtype`** (`int`): Optional. The default value is `None`. Output data type. Valid values: 5 indicates `torch.float16`, 6 indicates `torch.float32`, and 15 indicates `torch.bfloat16`.

    <!-- npu="910b" id16 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id16 -->

- **`x1_dtype`** (`int`): Optional. The default value is `None`. Input data type. When the data type of the `x1` tensor is `torch_npu.hifloat8`, `torch_npu.hifloat8` must be passed; when the type is `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e2m1fn_x2` must be passed.

    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id17 -->

- **`x2_dtype`** (`int`): Optional. The default value is `None`. Input data type. When the data type of the `x2` tensor is `torch_npu.hifloat8`, `torch_npu.hifloat8` must be passed; when the type is `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e2m1fn_x2` must be passed.

    <!-- npu="910b" id18 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id18 -->

- **`dequant_scale_dtype`** (`int`): Optional. The default value is `None`. Input data type. When the data type of `dequant_scale` is `torch.float8_e8m0fnu`, the `torch.float8_e8m0fnu` type must be passed.

    <!-- npu="910b" id19 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id19 -->

- **`pertoken_scale_dtype`** (`int`): Optional. The default value is `None`. Input data type. When the data type of `pertoken_scale` is `torch.float8_e8m0fnu`, the `torch.float8_e8m0fnu` type must be passed.

    <!-- npu="910b" id20 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id20 -->

- **`comm_quant_mode`** (`int`): Optional. The default value is `0`. Quantization mode for low-bit communication. The value is 0 or 1. When dynamic quantization low-bit communication is required, the value must be 1.

    <!-- npu="910b" id21 -->
    - <term>Atlas A2 products</term>: **This parameter is not supported currently.**
    <!-- end id21 -->

- **`comm_mode`** (`str`): Optional. Communication engine mode. The default value is `None`.

    <!-- npu="910b" id22 -->
    - <term>Atlas A2 products</term>: The values `None` and `ai_cpu` are supported, and both use AI CPU communication.
    <!-- end id22 -->
    <!-- npu="950" id23 -->
    - <term>Ascend 950PR&950DT products</term>: The values `None`, `ai_cpu`, and `ccu` are supported. `ai_cpu` and `ccu` directly specify AI CPU and CCU communication; `None` uses AI CPU communication by default.
    <!-- end id23 -->

## Return Values

`Tensor`

In non-quantization and fake-quantization scenarios, the data type is identical to that of `x1`. In full quantization scenarios, the output data type can be `torch.float16` or `torch.bfloat16`. The size of the 0th dimension is identical to that of the 0th dimension of `x1`. When `x1` is 2D, the size of the 1st dimension is identical to that of the 1st dimension of `x2`. When `x1` is 3D, the size of the 1st dimension is identical to that of the 1st dimension of `x1`, and the size of the 2nd dimension is identical to that of the 1st dimension of `x2`.

## Constraints

- This API can be used in inference scenarios.
- This fused operator is disabled in incremental quantization scenarios and enabled in full quantization scenarios.

  <!-- npu="910b" id24 -->
  - <term>Atlas A2 products</term>: Host CPU+TS communication is supported.
  <!-- end id24 -->
  <!-- npu="950" id25 -->
  - <term>Ascend 950PR&950DT products</term>: CCU and AI CPU communication are supported. CCU supports only intra-node UB domain interconnection, while AI CPU supports inter-node UB domain interconnection.
  <!-- end id25 -->

- This API supports graph mode.
- The input `x1` can be 2D or 3D with shape `(m, k)` or `(b, s, k)`, and `x2` must be 2D with shape `(k, n)`. The $k$ dimension must match and meet the requirements of the mm operator. The parameter `bias` supports only 1D tensors, where the size must be identical to that of the last dimension of `output`. The shape of `x3` must be identical to that of `output`.
- `x1` does not support transposed input. If `x2` is transposed, the size of its first dimension must match the last dimension of `x1`, satisfying the requirements of the MatMul operation.
- The value range of $k$ in `antiquant_group_size` is identical to that in the MatMul operation, `[1, 65535]`. `INT_MAX` must be greater than $(k-1)$.

<!-- npu="910b" id26 -->
- <term>Atlas A2 products</term>:
  - The data type can be `torch.bfloat16`.
  - Empty tensors are not supported for `x1` and `x2`.
  - Configurations of 1, 2, 4, and 8 ranks are supported. Only all-mesh networking over HCCS links is supported.
  - The shapes of `comm_quant_scale_1` and `comm_quant_scale_2` must be identical, and their `dtype` must be identical to the output `dtype`. These parameters are supported only in full quantization scenarios.
<!-- end id26 -->
<!-- npu="950" id27 -->
- <term>Ascend 950PR&950DT products</term>:
  - Non-quantization scenarios: The scenario where k is 0 is supported, and the output is bias + x3. bs/m/n of 0 is supported; in this case, the output passed in must also be an empty tensor. In this scenario, kernel computation is skipped and the result is returned directly.
  - Full quantization scenarios: Empty tensors are not supported.
  - Fake-quantization scenarios: Only empty tensors whose k axis is 0 are supported.
  - Only 1, 2, 4, 8, 16, 32, and 64 ranks are supported. In CCU mode, 1 rank is not supported.
<!-- end id27 -->
- Non-quantization scenarios: The values of b*s, m, k, and n must not exceed 2147483647 (INT32_MAX).
- Full quantization scenarios: The value range of $m$ is [1, 2147483647], and the size of the last dimension for `x1` and `x2` is [1, 65535]. Therefore, the value range of $k$ is [1, 65535]. The value of $n$ can be greater than 65535 only when `x2` with shape `[n, k]` is transposed.
- Fake-quantization scenarios: The value range of $m$ is [1, 2147483647], and the value range of $k$ and $n$ is [1, 65535].
- The computation-communication fusion MC2 operators within a single model support only the same communication domain.

<!-- npu="910b" id28 -->
- <term>Atlas A2 products</term>: The computation-communication fusion operators (AllGatherMatmul, MatmulReduceScatter, and MatmulAllReduce) within a single model support only the same communication domain.
<!-- end id28 -->
- In long-sequence scenarios, memory insufficiency or calculation timeout errors can occur as the size of $b/s$ or $m$ increases.
- The data type varies in different scenarios.

    **Table 1** Non-quantization scenarios

    | x1 | x2 | bias | x3 | output | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None | None |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None | None | None |

    <!-- npu="910b" id29 -->
    **Table 2** Fake-quantization scenarios (for <term>Atlas A2 products</term>)

    | x1 | x2 | bias | x3 | output | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.int8` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None |
    | `torch.bfloat16` | `torch.int8` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None |
    <!-- end id29 -->

    <!-- npu="950" id30 -->
    **Table 3** Fake-quantization scenarios (for <term>Ascend 950PR&950DT products</term>)

    | x1 | x2 | bias | x3 | output | antiquant_scale | antiquant_offset | dequant_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch_npu.hifloat8`/`torch.float8_e4m3fn` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | None | None | None | None |
    | `torch.bfloat16` | `torch_npu.hifloat8`/`torch.float8_e4m3fn` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | None | None | None | None |
    <!-- end id30 -->

    <!-- npu="910b" id31 -->
    **Table 4** Full quantization scenarios (for <term>Atlas A2 products</term>)

    | x1 | x2 | bias | x3 | output | antiquant_scale | antiquant_offset | dequant_scale | pertoken_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float16` | `torch.float16` | None | None | `torch.uint64`/`torch.int64` | None | None/`torch.float16` | None/`torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.bfloat16` | None | None | `torch.bfloat16` | None | None/`torch.bfloat16` | None/`torch.bfloat16` |
    <!-- end id31 -->

    <!-- npu="950" id32 -->
    **Table 5** Full quantization scenarios (for <term>Ascend 950PR&950DT products</term>)

    | x1 | x2 | bias | x3 | output | antiquant_scale | antiquant_offset | dequant_scale | pertoken_scale | comm_quant_scale_1 | comm_quant_scale_2 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float16` | `torch.float16` | None | None | `torch.float32` | `torch.float32` | None/`torch.float16` | None/`torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.bfloat16` | None | None | `torch.bfloat16` | `torch.float32` | None/`torch.bfloat16` | None/`torch.bfloat16` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float32` | `torch.float32` | None | None |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float32` | `torch.float32` | None | None |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None |
    | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.float32` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32`/`torch.float16`/`torch.bfloat16` | None | None | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | None | None |

    > [!NOTE]
    > In full quantization scenarios, if `dequant_scale` is provided as a `torch.float32` value, convert it into `torch.int64` by using the `torch_npu.npu_trans_quant_param` API before calling `torch_npu.npu_mm_all_reduce_base`. For details about the conversion method, see the corresponding API documentation.
    <!-- end id32 -->

- In full quantization scenarios, the dtype and shape relationships of `x1`, `x2`, `dequant_scale`, `pertoken_scale`, and groupSize in different quantization scenarios are as follows.

    | Quantization Type | x1 | x2 | dequant_scale | pertoken_scale | [groupSizeM,groupSizeN,groupSizeK] | Constraint |
    | --- | --- | --- | --- | --- | --- | --- |
    | perblock quantization | (b,s,k) | (k,n)/(n,k) | (ceilDiv(k,128),ceilDiv(n,128))/(ceilDiv(n,128),ceilDiv(k,128)) | (b,ceilDiv(m,128),ceilDiv(k,128)) | [128,128,128] | - |
    | perblock quantization | (m,k) | (k,n)/(n,k) | (ceilDiv(k,128),ceilDiv(n,128))/(ceilDiv(n,128),ceilDiv(k,128)) | (ceilDiv(m,128),ceilDiv(k,128)) | [128,128,128] | - |
    | MXFP quantization | (b,s,k) | (n,k) | (n,ceilDiv(k,64),2) | (s,ceilDiv(k,64),2) | [1,1,32] | When the `x1` and `x2` input types are `torch_npu.float4_e2m1fn_x2`, ceilDiv(k, 32) must be an even number. |
    | MXFP quantization | (m,k) | (n,k) | (n,ceilDiv(k,64),2) | (m,ceilDiv(k,64),2) | [1,1,32] | When the `x1` and `x2` input types are `torch_npu.float4_e2m1fn_x2`, ceilDiv(k, 32) must be an even number. |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op='sum')
        print("output: ", output)

    if __name__ == "__main__":
        worksize = 8
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype), nprocs=worksize)
    ```

- Graph mode call

    Call examples for the NZ data layout in non-quantization, fake-quantization, and full quantization scenarios:

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np
    class MM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
            output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                          x2=x2,
                                                          hcom=hcom,
                                                          reduce_op=reduce_op,
                                                          bias=bias,
                                                          antiquant_scale=antiquant_scale,
                                                          antiquant_offset=antiquant_offset,
                                                          x3=x3,
                                                          dequant_scale=dequant_scale
                                                          )
            return output_npu

    class MM_ALLREDUCE_A8W8_GRAPH_Model(MM_ALLREDUCE_GRAPH_Model):
        def __init__(self):
            super().__init__()
        def forward(self, x1, x2, hcom, reduce_op, bias, antiquant_scale, antiquant_offset, x3, dequant_scale):
            output_npu = torch_npu.npu_mm_all_reduce_base(x1=x1,
                                                          x2=x2.t(),
                                                          hcom=hcom,
                                                          reduce_op=reduce_op,
                                                          bias=bias,
                                                          antiquant_scale=antiquant_scale,
                                                          antiquant_offset=antiquant_offset,
                                                          x3=x3,
                                                          dequant_scale=dequant_scale
                                                          )
            return output_npu

    def define_model(model, graph_type):
        import torchair
        if graph_type == 1:  # Traditional graph input mode, static shape + online compilation
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN graph input mode, dynamic shape + binary file.
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model

    def get_graph(input, weight, hcomm_info, dequant_scale, bias, antiquant_scale, antiquant_offset, x3):
        model = MM_ALLREDUCE_A8W8_GRAPH_Model()
        model = define_model(model, 2)  # 1: Static graph input; 2: Dynamic graph input
        output = model(x1=input, x2=weight, hcom=hcomm_info, reduce_op="sum", bias=bias, antiquant_scale=antiquant_scale,
                       antiquant_offset=antiquant_offset, x3=x3, dequant_scale=dequant_scale)
        return output

    def run_mc2_a16w16(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.float16)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        output_a16w16 = get_graph(input, weight, hcom_info, None, None, None, None, None)
        return output_a16w16

    def run_mc2_a8w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.int8)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        # weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        dequant_scale = torch.randn(x2_shape[0], dtype=torch.float32).uniform_(float(-10), float(10)).npu()
        dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
        output_a8w8 = get_graph(input, weight, hcom_info, dequant_scale, None, None, None, None)
        return output_a8w8

    def run_mc2_a16w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        # weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        antiquant_scale = torch.randn(x2_shape[0], dtype=torch.float16).uniform_(float(-1), float(1)).npu()
        antiquant_offset = torch.ones(x2_shape[0], dtype=torch.float16).npu()
        output_a16w8 = get_graph(input, weight, hcom_info, None, None, antiquant_scale, antiquant_offset, None)
        return output_a16w8

    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, op_type):
        torch_npu.npu.set_device(rank)
        init_method = 'tcp://' + master_ip + ':' + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        output = None
        # Non-quantization call
        if op_type == "a16w16":
            output = run_mc2_a16w16(x1_shape, x2_shape, hcom_info)
        # Fake-quantization call
        if op_type == "a16w8":
            output = run_mc2_a16w8(x1_shape, x2_shape, hcom_info)
        # Full-quantization call
        if op_type == "a8w8":
            output = run_mc2_a8w8(x1_shape, x2_shape, hcom_info)
        print("output:", output)
    if __name__ == "__main__":
        worksize = 2
        master_ip = '127.0.0.1'
        master_port = '50001'
        x1_shape = [1280, 5120]
        x2_shape = [640, 5120]
        op_type = "a16w16" # Options: a16w16, a16w8, a8w8
        mp.spawn(run_mm_all_reduce_base, args=(worksize, master_ip, master_port, x1_shape, x2_shape, op_type), nprocs=worksize)
    ```
