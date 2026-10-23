# torch_npu.npu_weight_quant_preprocess

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

This API preprocesses the fake quantization parameters of Matmul operators. The following data flows are supported:

- MM_MX_A8W4 data flow: MX quantization scenario of the QuantBatchMatmulV4 operator. A8W4 indicates that the left matrix is `torch.float8_e4m3fn` and the right matrix is `torch_npu.float4_e2m1fn_x2`.
- GMM_MX_A8W4 data flow: MX quantization scenario of the GroupedMatmul operator. A8W4 indicates that the left matrix is `torch.float8_e4m3fn` and the right matrix is `torch_npu.float4_e2m1fn_x2`.
- MM_A16S4 data flow: pertensor/perchannel/pergroup quantization scenario of the WeightQuantBatchMatmulV2 operator. A16S4 indicates that the left matrix is `torch.float16`/`torch.bfloat16` and the right matrix is `torch_npu.int4`.
- MM_A16F4 data flow: pergroup quantization scenario of the WeightQuantBatchMatmulV2 operator. A16F4 indicates that the left matrix is `torch.float16`/`torch.bfloat16` and the right matrix is `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`).
- MM_MX_A16F4 data flow: MX quantization scenario of the WeightQuantBatchMatmulV2 operator. A16F4 indicates that the left matrix is `torch.float16`/`torch.bfloat16` and the right matrix is `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`).

## Prototype

```python
torch_npu.npu_weight_quant_preprocess(weight, weight_scale, x_dtype, weight_dtype, weight_scale_dtype, weight_offset=None, bias=None, x_scale_dtype=None, k_group_size=0) -> Tuple[Tensor, Tensor, Tensor, Tensor]
```

## Parameters

- **`weight`** (`Tensor`): Required. Weight matrix of the Matmul. Non-contiguous `Tensor` is supported.
  - The logical data type can be `torch_npu.int4` (carried by `torch.uint8`) or `torch_npu.float4_e2m1fn_x2` (carried by `torch.uint8`). The data layout can be ND. 2D or 3D input is supported, and the logical shapes are `(K, N)` and `(G, K, N)` respectively, where $G$ indicates the G axis of the GroupedMatmul operator. One `torch.uint8` element packs two 4-bit values, and the number of elements in the packed dimension must be even. When packing along the K dimension, the physical K dimension length of `weight` is $K/2$. When packing along the N dimension, the physical N dimension length is $N/2$.
- **`weight_scale`** (`Tensor`): Required. Dequantization scale parameter of the weight. Non-contiguous `Tensor` is supported.
  - The logical data type can be `torch.float8_e8m0fnu` (carried by `torch.uint8`), `torch.float16`, or `torch.bfloat16`. The data layout can be ND or NCL.
    - MX quantization scenario: The A8W4 data flow supports 3D or 4D input, with shapes `(ceil_div(K, 64), N, 2)` and `(G, ceil_div(K, 64), N, 2)` respectively, where $G$ indicates the G axis of the GroupedMatmul operator. The A16F4 data flow supports 2D input with shape `(ceil_div(K, 32), N)`.
    - pertensor quantization scenario: The shape is `(1)` or `(1, 1)`.
    - perchannel quantization scenario: The shape is `(N)` or `(1, N)`.
    - pergroup quantization scenario: The shape is `(G, N)`, where $G$ is the number of groups.
- **`x_dtype`** (`int`): Required. Data type of the activation matrix of the Matmul.
  - The A16S4/A16F4 data flow supports `torch.float16` and `torch.bfloat16`. The A8W4 data flow supports `torch.float8_e4m3fn`.
- **`weight_dtype`** (`int`): Required. Specifies the data type actually carried in `weight`.
  - The value is `torch_npu.int4` for the A16S4 data flow and `torch_npu.float4_e2m1fn_x2` for the other data flows.
- **`weight_scale_dtype`** (`int`): Required. Specifies the data type actually carried in `weight_scale`. The value must be consistent with the actual data type of `weight_scale`.
  - `torch.float8_e8m0fnu`, `torch.float16`, and `torch.bfloat16` are supported.
- **`weight_offset`** (`Tensor`): Optional. Dequantization offset parameter of the weight. The default value is `None`.
  - The A16S4 data flows support pass-through, and their shapes and data types must be consistent with those of `weight_scale`. For the other data flows, pass `None`.
- **`bias`** (`Tensor`): Optional. Bias matrix of the Matmul. It must be a contiguous `Tensor`.
  - The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. 1D or 2D input is supported, with shape `(N)`, `(1, N)`, or `(G, N)`.
- **`x_scale_dtype`** (`int`): Optional. Data type of the activation quantization scale parameter. The default value is `None`.
  - The A8W4 data flow supports only `torch.float8_e8m0fnu`. For the A16S4/A16F4 data flows, pass `None`.
- **`k_group_size`** (`int`): Optional. Group size of the weight in the K dimension during pergroup quantization. The default value is `0`.
  - The value is `32` in the MX quantization scenario of the A8W4/A16F4 data flows. The value is an integer greater than `0` in the pergroup quantization scenario of the A16S4/A16F4 data flows. In other scenarios, the default value `0` is used.

## Return Values

`Tuple[Tensor, Tensor, Tensor, Tensor]`

Four `Tensor` values are returned:

- **`out_weight`** (`Tensor`): Preprocessed `weight`. The output data layout in different scenarios is as follows:
  - In the A16S4 data flow, for the pertensor (`weight` transposed/non-transposed), perchannel (`weight` transposed), and pergroup (`weight` transposed) quantization scenarios, the data layout of `weight` remains ND.
  - In the A16S4 data flow, for the perchannel (`weight` non-transposed) and pergroup (`weight` non-transposed) quantization scenarios, the data layout of `weight` changes from ND to FRACTAL_NZ_C0_8.
  - In the A8W4 data flow, for the MX quantization scenario, the data layout of `weight` changes from ND to FRACTAL_NZ_C0_16.
  - In the A16F4 data flow, for the pergroup quantization scenario (only `weight` non-transposed is supported), the data layout of `weight` changes from ND to FRACTAL_NZ_C0_8.
  - In the A16F4 data flow, for the MX quantization scenario, when `weight` is a transposed input, the output is a direct copy in ND format and keeps the same sizes/strides as the input. When the input is non-transposed, the layout changes from ND to FRACTAL_NZ_C0_8.
- **`out_weight_scale`** (`Tensor`): Preprocessed `weight_scale`. The data type is the same as that of the input `weight_scale`.
- **`out_weight_offset`** (`Tensor`): Preprocessed `weight_offset`. When `weight_offset` is not passed, an empty `Tensor` is returned. For the A16S4 data flow, when it is passed, it is passed through and keeps the same sizes/strides as the input.
- **`out_bias`** (`Tensor`): Preprocessed `bias`. The data type is the same as that of the input `bias` (if `bias` is provided).

## Constraints

The following parameter combinations are currently supported:

- **MM_MX_A8W4 data flow**:
  - The data type of `weight` must be `torch_npu.float4_e2m1fn_x2`, the data layout must be ND, and K must satisfy $K \% k\_group\_size = 0$.
  - The logical shape of `weight` is `{K, N}`. When `torch.uint8` is used to carry fp4 data, the view shape is `{K/2, N}` and the storage shape is `{N, K/2}` (transposed), with stride `[1, K/2]`.
  - The data type of `weight_scale` must be `torch.float8_e8m0fnu`, and the data layout must be ND/NCL.
  - The view shape of `weight_scale` is `{ceil_div(K, 64), N, 2}`, and the storage shape is `{N, ceil_div(K, 64), 2}` (transposed).
  - `k_group_size` must be `32`.
  - `x_dtype` must be `torch.float8_e4m3fn`.
  - `x_scale_dtype` must be `torch.float8_e8m0fnu`.
  - `weight_offset` is not supported currently and must be passed as `None`.
- **GMM_MX_A8W4 data flow**:
  - The data type of `weight` must be `torch_npu.float4_e2m1fn_x2`, the data layout must be ND, and K must satisfy $K \% k\_group\_size = 0$.
  - The logical shape of `weight` is `{G, K, N}`. When `torch.uint8` is used to carry fp4 data, the view shape is `{G, K/2, N}` and the storage shape is `{G, N, K/2}` (transposed), with stride `[K * N/2, 1, K/2]`.
  - The data type of `weight_scale` must be `torch.float8_e8m0fnu`, and the data layout must be ND/NCL.
  - The view shape of `weight_scale` is `{G, ceil_div(K, 64), N, 2}`, and the storage shape is `{G, N, ceil_div(K, 64), 2}` (transposed), with stride `[N * ceil_div(K, 64) * 2, 2, ceil_div(K, 64) * 2, 1]`.
  - `k_group_size` must be `32`.
  - `x_dtype` must be `torch.float8_e4m3fn`.
  - `x_scale_dtype` must be `torch.float8_e8m0fnu`.
  - `weight_offset` is not supported currently and must be passed as `None`.
- **MM_A16S4 data flow**:
  - The data type of `weight` must be `torch_npu.int4`, carried by `torch.uint8` (one `torch.uint8` element packs two `torch_npu.int4` values). The data layout must be ND. 2D input is supported, and both transposed and non-transposed inputs are supported.
  - When not transposed, the shape of `weight` is `{K, N/2}` (packed along the N dimension). When transposed, the shape is `{K/2, N}` (packed along the K dimension). The number of elements in the packed dimension must be even.
  - The data type of `weight_scale` is `torch.float16` or `torch.bfloat16`, and its shape determines the quantization granularity.
  - `x_dtype` must be `torch.float16` or `torch.bfloat16`.
  - `x_scale_dtype` must be passed as `None`.
  - `weight_offset` supports pass-through, and its shape and data type must be consistent with those of `weight_scale`.
  - The following three scenarios are classified according to the shape of `weight_scale`:
    - pertensor: `weight_scale` contains only one element, with shape `{1}` or `{1, 1}`. `k_group_size` uses the default value `0`. `out_weight` is a direct ND copy and keeps the same sizes/strides as the input `weight`.
    - perchannel: The shape of `weight_scale` is `{N}` or `{1, N}`. `k_group_size` uses the default value `0`. When the input is transposed, `out_weight` is a direct ND copy. When the input is not transposed, `out_weight` is FRACTAL_NZ_C0_8 output, with storage shape `{ceil_div(N, 16), ceil_div(K, 16), 16, 8}` (N blocks first).
    - pergroup: The shape of `weight_scale` is `{G, N}`, where $G$ is the number of groups and $G > 1$. `k_group_size` must be greater than `0`, indicating the group size per group in the K dimension. The output routing is the same as that in the perchannel scenario. When `weight` is transposed input, `weight_scale` must have the same layout as `weight` (both are transposed views).
- **MM_A16F4 data flow**:
  - The data type of `weight` must be `torch_npu.float4_e2m1fn_x2`, carried by `torch.uint8` (one `torch.uint8` element packs two fp4 values). The data layout must be ND. 2D input is supported, and only non-transposed is supported, with shape `{K, N/2}` (packed along the N dimension). The number of elements in the packed dimension must be even.
  - The data type of `weight_scale` is `torch.float16` or `torch.bfloat16`, and the shape is `{G, N}`, where $G$ is the number of groups and $G > 1$.
  - `k_group_size` must be greater than `0`, indicating the group size per group in the K dimension.
  - `x_dtype` must be `torch.float16` or `torch.bfloat16`.
  - `x_scale_dtype` must be passed as `None`.
  - `weight_offset` is not supported currently and must be passed as `None`.
  - `out_weight` is FRACTAL_NZ_C0_8 output, with storage shape `{ceil_div(N, 16), ceil_div(K, 16), 16, 8}` (N blocks first).
- **MM_MX_A16F4 data flow**:
  - The data type of `weight` must be `torch_npu.float4_e2m1fn_x2`, carried by `torch.uint8` (one `torch.uint8` element packs two fp4 values). The data layout must be ND. 2D input is supported, and both transposed and non-transposed inputs are supported.
  - When not transposed, the shape of `weight` is `{K, N/2}` (packed along the N dimension). When transposed, the shape is `{K/2, N}` (packed along the K dimension), with stride `[1, K/2]`. The number of elements in the packed dimension must be even, and K must satisfy $K \% 32 = 0$.
  - The data type of `weight_scale` must be `torch.float8_e8m0fnu`, and the shape is `{ceil_div(K, 32), N}`. When `weight` is transposed input, `weight_scale` must have the same layout as `weight` (both are transposed views, with stride `[1, ceil_div(K, 32)]`).
  - `k_group_size` must be `32`.
  - `x_dtype` must be `torch.float16` or `torch.bfloat16`.
  - `x_scale_dtype` must be passed as `None`.
  - `weight_offset` is not supported currently and must be passed as `None`.

## Examples

- MM_MX_A8W4 data flow scenario

  ```python
  import torch
  import torch_npu

  # MM_MX_A8W4 data flow example
  k = 64  # Logical K dimension length
  n = 128
  packed_k = k // 2
  k_scale = (k + 63) // 64

  # weight: float4_e2m1. Each uint8 packs two fp4 values.
  # logical shape: {K, N}, view: {K/2, N}, storage: {N, K/2}
  cpu_weight = torch.randint(0, 255, (n, packed_k), dtype=torch.uint8)
  weight = cpu_weight.npu().transpose(0, 1)

  # weight_scale: float8_e8m0, 3-D transposed
  # view: {ceil_div(K, 64), N, 2}, storage: {N, ceil_div(K, 64), 2}
  cpu_scale = torch.randint(0, 255, (n, k_scale, 2), dtype=torch.uint8)
  weight_scale = cpu_scale.npu().transpose(0, 1)

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float8_e4m3fn,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float8_e8m0fnu,
      weight_offset=None,
      bias=None,
      x_scale_dtype=torch.float8_e8m0fnu,
      k_group_size=32
  )
  ```

- GMM_MX_A8W4 data flow scenario

  ```python
  import torch
  import torch_npu

  # GMM_MX_A8W4 data flow example
  g = 4
  k = 64  # Logical K dimension length
  n = 128
  packed_k = k // 2
  k_scale = (k + 63) // 64

  # weight: float4_e2m1. Each uint8 packs two fp4 values.
  # logical shape: {G, K, N}, view: {G, K/2, N}, storage: {G, N, K/2}
  cpu_weight = torch.randint(0, 255, (g, n, packed_k), dtype=torch.uint8)
  weight = cpu_weight.npu().transpose(1, 2)

  # weight_scale: float8_e8m0, 4-D transposed
  # view: {G, ceil_div(K, 64), N, 2}, storage: {G, N, ceil_div(K, 64), 2}
  cpu_scale = torch.randint(0, 255, (g, n, k_scale, 2), dtype=torch.uint8)
  weight_scale = cpu_scale.npu().transpose(1, 2)

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float8_e4m3fn,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float8_e8m0fnu,
      weight_offset=None,
      bias=None,
      x_scale_dtype=torch.float8_e8m0fnu,
      k_group_size=32
  )
  ```

- MM_A16S4 data flow scenario (default perchannel, `weight` non-transposed, output FRACTAL_NZ_C0_8)

  ```python
  import torch
  import torch_npu
  import numpy as np

  # MM_A16S4 data flow example
  # Switches: quantization type and weight transpose state
  quant_mode = "perchannel"  # The value can be "pertensor", "perchannel", or "pergroup".
  weight_trans = False       # True indicates transposed weight input.

  k, n = 256, 128
  k_group_size = 64 if quant_mode == "pergroup" else 0

  # weight: int4. Each uint8 packs two int4 values.
  weight_int4 = torch.randint(low=-8, high=8, size=(k, n), dtype=torch.int8)
  w = (weight_int4.numpy() & 0xF).astype(np.uint8)
  if weight_trans:
      # Transposed input: packed along the K dimension. logical shape {K, N}, physical shape {N, K/2}
      # Transposed view shape {K/2, N}, stride [1, K/2]
      packed = np.ascontiguousarray((w[1::2, :].T << 4) | w[0::2, :].T)
      weight = torch.from_numpy(packed).npu().transpose(0, 1)
  else:
      # Non-transposed input: packed along the N dimension. logical shape {K, N}, packed shape {K, N/2}
      weight = torch.from_numpy((w[:, 1::2] << 4) | w[:, 0::2]).npu()

  # weight_scale: float16. The shape depends on the quantization type.
  if quant_mode == "pertensor":
      weight_scale = torch.randn((1,), dtype=torch.float16).npu()
  elif quant_mode == "perchannel":
      weight_scale = torch.randn((n,), dtype=torch.float16).npu()
  else:  # pergroup
      g = (k + k_group_size - 1) // k_group_size
      weight_scale = torch.randn((g, n), dtype=torch.float16)
      if weight_trans:
          # For transposed input, weight_scale must have the same layout as weight (both are transposed views).
          weight_scale = weight_scale.t().contiguous().t()
      weight_scale = weight_scale.npu()

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float16,
      weight_dtype=torch_npu.int4,
      weight_scale_dtype=torch.float16,
      k_group_size=k_group_size
  )
  ```

- MM_A16F4 data flow scenario (pergroup, `weight` non-transposed, output FRACTAL_NZ_C0_8)

  ```python
  import torch
  import torch_npu

  # MM_A16F4 data flow example
  k, n = 256, 128
  k_group_size = 64
  g = (k + k_group_size - 1) // k_group_size

  # weight: float4_e2m1. Each uint8 packs two fp4 values.
  # Non-transposed input: packed along the N dimension. logical shape {K, N}, packed shape {K, N/2}
  weight = torch.randint(0, 255, (k, n // 2), dtype=torch.uint8).npu()

  # weight_scale: float16, pergroup shape {G, N}
  weight_scale = torch.randn((g, n), dtype=torch.float16).npu()

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float16,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float16,
      k_group_size=k_group_size
  )
  ```

- MM_MX_A16F4 data flow scenario

  ```python
  import torch
  import torch_npu

  # MM_MX_A16F4 data flow example
  # Switch: weight transpose state
  weight_trans = False  # True indicates transposed weight input.

  k, n = 256, 128
  g = k // 32  # k_group_size=32, number of groups in the K dimension of scale

  # weight: float4_e2m1. Each uint8 packs two fp4 values.
  # weight_scale: float8_e8m0, carried by uint8, shape {K/32, N}
  if weight_trans:
      # Transposed input: weight is packed along the K dimension. view shape {K/2, N}, stride [1, K/2]
      # For transposed input, weight_scale must have the same layout as weight (both are transposed views).
      weight = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8).npu().transpose(0, 1)
      weight_scale = torch.randint(0, 255, (n, g), dtype=torch.uint8).view(
          torch.float8_e8m0fnu).npu().transpose(0, 1)
  else:
      # Non-transposed input: weight is packed along the N dimension. shape {K, N/2}
      weight = torch.randint(0, 255, (k, n // 2), dtype=torch.uint8).npu()
      weight_scale = torch.randint(0, 255, (g, n), dtype=torch.uint8).view(
          torch.float8_e8m0fnu).npu()

  out_weight, out_weight_scale, out_weight_offset, out_bias = torch_npu.npu_weight_quant_preprocess(
      weight,
      weight_scale,
      x_dtype=torch.float16,
      weight_dtype=torch_npu.float4_e2m1fn_x2,
      weight_scale_dtype=torch.float8_e8m0fnu,
      k_group_size=32
  )
  ```
