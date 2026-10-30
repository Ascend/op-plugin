# torch_npu.npu_grouped_matmul_swiglu_quant_v2

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

## Function

- API function: `torch_npu.npu_grouped_matmul_swiglu_quant_v2` is a computation method that fuses grouped matrix multiplication (GroupedMatmul), the SwiGLU activation function, and quantization (quant). This method is applicable to scenarios where the matrix multiplication result needs to be activated by the SwiGLU function. The fused operator can parallelize part of the process at the underlying level to achieve performance optimization.

- Formulas

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    <details>
    <summary>A8W8 quantization scenarios (A indicates the activation matrix, W indicates the weight matrix, and 8 indicates the `torch.int8` data type):</summary>

    - **Inputs**
      * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
      * $W∈\mathbb{Z_8}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
      * $w\_scale∈\mathbb{R}^{E \times N}$: channel-wise scaling factor of the grouped weight matrix.
      * $x\_scale∈\mathbb{R}^{M}$: token-wise scaling factor of the activation matrix.
      * $groupList∈\mathbb{N}^{E}$: group index list in `cumsum` or `count` form.
    - **Outputs**
      * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation**
      1. Determine the token range for each group based on `groupList[i]`, where $i \in [0,Len(groupList)]$.
      2. Perform the following computations based on the inputs determined by grouping.

          $$
          C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

      3. Quantize the output results.

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>

    <details>
    <summary>A8W4 quantization scenarios (MSD) (A indicates the activation matrix, W indicates the weight matrix, 8 indicates the `torch.int8` data type, and 4 indicates the `torch_npu.int4` data type):</summary>

    - **Inputs**
      * $X∈\mathbb{Z_8}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
      * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
      * $weightAssistMatrix∈\mathbb{R}^{E \times N}$: auxiliary matrix used for matrix multiplication, which is generated offline and not within the operator.
      * $w\_scale$: scaling factor of the grouped weight matrix. In `perchannel` mode, its shape is $\mathbb{R}^{E \times N}$. In `pergroup` mode, its shape is $\mathbb{R}^{E \times K\_group\_num \times N}$.
      * $x\_scale∈\mathbb{R}^{M}$: token-wise scaling factor of the activation matrix.
      * $groupList∈\mathbb{N}^{E}$: group index list in `cumsum` or `count` form.
    - **Outputs**
      * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation**
      1. Determine the token range for each group based on `groupList[i]`, using identical grouping logic to that of A8W8.
      2. Split the `torch.int8` left matrix input into high and low 4-bit parts.

          $$
          X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor
          $$

          $$
          X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8
          $$

      3. Perform matrix multiplication separately for the high and low parts, apply `perchannel` or `pergroup` quantization scaling, and combine the results with the auxiliary matrix.

          $$
          C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          $$
          Swish(x)=\frac{x}{1+e^{-x}}
          $$

      4. Quantize the output results.

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>

    <details>
    <summary>A4W4 quantization scenarios (A indicates the activation matrix, W indicates the weight matrix, and 4 indicates the `torch_npu.int4` data type):</summary>

    - **Inputs**
      * $X∈\mathbb{Z_4}^{M \times K}$: activation matrix (left matrix), where $M$ indicates the total number of tokens and $K$ indicates the feature dimension.
      * $W∈\mathbb{Z_4}^{E \times K \times N}$: grouped weight matrix (right matrix), where $E$ indicates the number of routed experts, $K$ indicates the feature dimension, and $N$ indicates the output dimension.
      * $w\_scale∈\mathbb{R}^{E \times N}$: channel-wise scaling factor of the grouped weight matrix.
      * $x\_scale∈\mathbb{R}^{M}$: token-wise scaling factor of the activation matrix.
      * $smoothScale∈\mathbb{R}^{E \times N/2}$: smooth scaling factor, where $E$ indicates the number of routed experts and $N$ indicates the output dimension. Broadcasting is supported when the shape is `(E,)`.
      * $groupList∈\mathbb{N}^{E}$: group index list in `cumsum` or `count` form.
    - **Outputs**
      * $Q∈\mathbb{Z_8}^{M \times N/2}$: quantized output matrix.
      * $Q\_scale∈\mathbb{R}^{M}$: quantization scale factor.
    - **Computation**
      1. Determine the token range for each group based on `groupList[i]`, using identical grouping logic to that of A8W8.
      2. Perform the following computations based on the inputs determined by grouping.

          $$
          C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
          $$

          $$
          C_{i,act}, gate_{i} = split(C_{i})
          $$

          $$
          S_{i}=Swish(C_{i,act})\odot gate_{i}
          $$

          $$
          Swish(x)=\frac{x}{1+e^{-x}}
          $$

          $$
          S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}
          $$

          Note: When the shape of `smoothScale` is `(E,)`, it is broadcasted to match the shape of $S_{i}$.

      3. Quantize the output results.

          $$
          Q\_scale_{i} = \frac{max(|S_{i}|)}{127}
          $$

          $$
          Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
          $$

    </details>
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>:
    <details>
    <summary>MX quantization scenarios:</summary>

      1. Perform the following computation using the inputs determined based on the grouping:

         $$
         C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}
         $$

         $$
         C_{i,act}, gate_{i} = split(C_{i})
         $$

         $$
         S_{i}=Swish(C_{i,act})\odot gate_{i}
         $$

         $$
         Swish(x)=\frac{x}{1+e^{-x}}
         $$

      2. Quantize the output:

         $$
         shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax
         $$

         $$
         Q\_scale = 2 ^ {shared\_exp}
         $$

         $$
         Q_i = quantize\_to\_element\_format(S_i/Q\_scale), \space i\space from\space 1\space to\space blocksize
         $$

         where $emax$ denotes the exponent of the maximum normal value for the corresponding data type.

         |   DataType    | emax |
         | :-----------: | :--: |
         | `torch.float8_e4m3fn` |  8   |
         | `torch.float8_e5m2`  |  15  |
         | `torch.float4_e2m1fn_x2`  |  2   |

         where $blocksize$ denotes the number of elements per quantization block. Only `32` is supported.

    </details>

    <details>
    <summary>pertoken quantization scenarios:</summary>

      - **Definition**:
        - **⋅** indicates matrix multiplication.
        - **⊙** indicates element-wise multiplication.
      - **Input**:
        - $X∈\mathbb{Z_8}^{M \times K}$: Activation matrix (left matrix), where M is the total number of tokens and K is the feature dimension.
        - $W∈\mathbb{Z_8}^{E \times K \times N}$: Grouped weight matrix (right matrix), where E is the number of experts, K is the feature dimension, and N is the output dimension.
        - $w_scale∈\mathbb{R}^{E \times N}$: Per-channel scaling factor of the weight matrix.
        - $x_scale∈\mathbb{R}^{M}$: Per-token scaling factor of the activation matrix.
        - $groupList∈\mathbb{N}^{E}$: Group index list in cumsum or count form.
      - **Output**:
        - $Q∈\mathbb{Z_8}^{M \times N / 2}$: Quantized output matrix.
        - $Q_scale∈\mathbb{R}^{M}$: Per-token quantization scaling factor of the output matrix.
      - **Computation process**:

      1. Determine the tokens of the current group based on groupList[i], where $i \in [0,Len(groupList)]$.

      2. Perform the following computation based on the input determined by the group:

         $$
         C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i} \odot w\_scale_{i}
         $$

         $$
         C_{i,act}, gate_{i} = split(C_{i})
         $$

         $$
         S_{i}=Swish(C_{i,act})\odot gate_{i}
         $$

         $$
         Swish(x)=\frac{x}{1+e^{-x}}
         $$

         where $x\_scale_{i}$ denotes the quantization factor for the corresponding token.

      3. Quantize the output result:

         $$
         Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}
         $$

         $$
         Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil
         $$

    </details>
  <!-- end id5 -->

## Prototype

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=6, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None, x_dtype=None, weight_dtype=None, weight_scale_dtype=None, x_scale_dtype=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Left matrix for matrix multiplication, $X$ in the formulas. This parameter must be 2D with shape `[m, k]`. The data layout can be ND. Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data type can be `torch.int8`.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch.float4_e2m1fn_x2`, `torch.int8`, `torch_npu.hifloat8`, or `torch_npu.float4_e1m2fn_x2` (supported only when `weight` is in $FRACTAL_NZ$ format). For the `torch_npu.hifloat8` and `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` series, the optional parameter `x_dtype` must be set to the corresponding type. In this case the `dtype` of the input `x` itself no longer takes effect, but the `dtype` of the input `x` itself must still be an 8-bit data type to ensure that the `shape` is correct. The inner axis `K` of float4 must be even so that 8-bit data can be converted into two float4 values. The data format supports $ND$.
  <!-- end id7 -->

- **`weight`** (`List[Tensor]`): Required. Weight matrix (right matrix of the matrix multiplication). Non-contiguous `Tensor`s are supported.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data type can be `torch.int8`.
    - When the data layout is ND, the shape supports 3 dimensions `[e, k, n]`.
    - When the data layout is FRACTAL_NZ (format conversion can be implemented through the `npu_format_cast` API), the shape supports 5 dimensions. Taking non-transposed as an example, in `torch.float8_e4m3fn` scenarios the shape is `[e, k/32, n/16, 16, 32]`, and in `torch.float4_e2m1fn_x2` scenarios the shape is `[e, k/64, n/16, 16, 64]`.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: A single `Tensor` (the `Tensor` list length must be 1) and multiple `Tensor`s (the `Tensor` list length is e) are supported.
    - When the data layout is `ND`, 3D shapes are supported: non-transposed shape `(e, k, n)` and transposed shape `(e, n, k)`. Supported data types are `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch.float4_e2m1fn_x2`, `torch.int8`, and `torch_npu.hifloat8`. For `torch_npu.hifloat8` and the `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` series, the optional parameter `weight_dtype` must be set to the corresponding data type. In this case, the data type of `weight` itself is ignored, but `weight` must still have an 8-bit data type to ensure the shape is correct. For the `float4` series, the inner dimension must be even so that 8 bits can be converted into two `float4` values.
    - When the data layout is `FRACTAL_NZ` (conversion can be performed using `npu_format_cast`), 5D shapes are supported: non-transposed shape `(e, n/32, k/16, 16, 32)` and transposed shape `(e, k/32, n/16, 16, 32)`. Only `torch.float8_e4m3fn`, `torch_npu.float4_e1m2fn_x2`, and `torch.float4_e2m1fn_x2` are supported.
  <!-- end id9 -->

- **`weight_scale`** (`TensorList`): Required. Quantization factor for the weight matrix,  $w_{scale}$ in the formulas. Currently, only a `TensorList` of length `1` is supported. The data layout can be ND. Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The `shape` supports 2D `[e, n]`, and the data type supports `torch.float32`.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: In MX quantization scenarios, this parameter can be 4D with shape `(e, ceil(k / 64), n, 2)` (non-transposed) or `(e, n, ceil(k / 64), 2)` (transposed); and the data type can be `torch.float8_e8m0fnu`. In `pertoken` quantization scenarios, this parameter can be 2D with shape `(e, n)`. When `x` is `torch.int8`, the data type of `weight_scale` can be `torch.bfloat16`, `torch.float32`, or `torch.float16`. When `x` is `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`, the data type of `weight_scale` can be `torch.bfloat16` or `torch.float32`.
  <!-- end id11 -->

- **`x_scale`** (`Tensor`): Required. Quantization factor for the activation matrix, $x\_scale$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: This parameter must be 1D with shape `(m)`. The data type can be `torch.float32`.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: In MX quantization scenarios, this parameter can be 3D with shape `(m, ceil(k / 64), 2)`. The data type can be `torch.float8_e8m0fnu`. In `pertoken` quantization scenarios, this parameter must be 1D with shape `(m)`. The data type can be `torch.float32`.
  <!-- end id13 -->

- **`*`**: Variables before `*` are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`group_list`** (`Tensor`): **Required.** Number of tokens in each group involved in the computation. The `shape` can be 1D `[e]`. The data type can be `torch.int64`, and the data layout can be ND. Non-contiguous tensors are supported. When `group_list_type` is `0`, the last value must not be greater than the first dimension of `x`. When `group_list_type` is `1`, the sum of all values must not be greater than the first dimension of `x`. The values in `group_list` constrain the valid part of the output data. Parts not specified in `group_list` are not updated.

- **`smooth_scale`** (`Tensor`): Optional. Smooth scaling factor, $smoothScale$ in the formulas. The data type can be `torch.float32`. The data layout can be ND. This parameter must be provided only in A4W4 scenarios, and its first axis length must be identical to the first axis dimension of `weight`. The shape of this parameter is `(E, N/2)` or `(E,)`. When shape `(E,)` is used, broadcast multiplication is applied. In other scenarios, the default value is `None`.

- **`weight_assist_matrix`** (`TensorList`): Optional. Auxiliary matrix for the right matrix, $weightAssistMatrix$ in the formulas. The data type can be `torch.float32`. The data layout can be ND. This parameter must be a 2D tensor. This parameter must be provided only in A8W4 scenarios, where the length of its first dimension must be identical to that of the first dimension of `weight`, and the length of its last dimension must be identical to that of the last dimension of `weight` when restored to the ND layout. In other scenarios, the default value is `None`.

- **`bias`** (`Tensor`): Optional. Offset value for matrix multiplication computation, $bias$ in the formulas. This parameter must be a 2D tensor. The data type can be `torch.int32`. Currently, only the default value `None` is supported.

- **`dequant_mode`** (`int`): Optional. Dequantization mode. This parameter is of type `int32` and has a default value of `0`. A value of `0` indicates `pertoken` quantization for the activation matrix and `perchannel` quantization for the weight matrix. A value of `1` indicates `pertoken` quantization for the activation matrix and `pergroup` quantization for the weight matrix. A value of `2` indicates MX quantization.

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: In A8W4 scenarios, `dequant_mode` can be `0` or `1`. In A8W8 and A4W4 scenarios, `dequant_mode` can only be `0`.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: Currently, only `0` and `2` are supported.
  <!-- end id15 -->

- **`dequant_dtype`** (`int`): Optional. Dequantization data type. This parameter is of type `int32`.

  <!-- npu="A3,910b" id16 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, only the default value `6` (indicating `torch.float32`) is supported.
  <!-- end id16 -->
  <!-- npu="950" id17 -->
  - <term>Ascend 950PR&950DT products</term>: The default value is `torch.float32`. Currently, `torch.float32`, `torch.bfloat16`, and `torch.float16` are supported.
  <!-- end id17 -->

- **`quant_mode`** (`int`): Optional. Quantization mode after SwiGLU. This parameter is of type `int32`. Valid values are `0` (default, `pertoken` quantization), `1` (`pergroup` quantization), or `2` (MX quantization).

  <!-- npu="A3,910b" id18 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, only the default value `0` (`pertoken` quantization) is supported.
  <!-- end id18 -->
  <!-- npu="950" id19 -->
  - <term>Ascend 950PR&950DT products</term>: Currently, only `0` and `2` are supported.
  <!-- end id19 -->

- **`quant_dtype`** (`int`): Optional. Low-bit data type after quantization. This parameter is of type `int32`.

  <!-- npu="A3,910b" id20 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, only the default value `0` (indicating `torch.int8`) is supported.
  <!-- end id20 -->
  <!-- npu="950" id21 -->
  - <term>Ascend 950PR&950DT products</term>: The default value is `torch.int8`. Currently, `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, `torch.int8`, and `torch_npu.hifloat8` are supported.
  <!-- end id21 -->

- **`group_list_type`** (`int`): Optional. Input type of `group_list`. This parameter is of type `int32` and has a default value of `0`.
  - A value of `0` indicates cumsum mode, where each element in `group_list` represents the cumulative length of the current group.
  - A value of `1` indicates count mode, where each element in `group_list` represents the number of elements in the corresponding group.

- **`tuning_config`** (`List[int]`): Optional. The first element in this parameter array specifies the expected number of tokens processed by each expert. Elements from the second element onward are reserved for future expansion and do not need to be specified by the user. The default value is `None`.

- **`x_dtype`** (`int`): Optional. Actual data type of the input `x`. Currently, only the default value `None` is supported, indicating that the actual data type of `x` is the same as its `dtype`.

  <!-- npu="A3,910b" id22 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, this parameter is not supported. Use the default value.
  <!-- end id22 -->
  <!-- npu="950" id23 -->
  - <term>Ascend 950PR&950DT products</term>: When `x` is `torch.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, or `torch_npu.hifloat8`, `x_dtype` must be set to `torch.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, or `torch_npu.hifloat8`, respectively.
  <!-- end id23 -->

- **`weight_dtype`** (`int`): Optional. Actual data type of the input `weight`. Currently, only the default value `None` is supported, indicating that the actual data type of `weight` is the same as its `dtype`.

  <!-- npu="A3,910b" id24 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, this parameter is not supported. Use the default value.
  <!-- end id24 -->
  <!-- npu="950" id25 -->
  - <term>Ascend 950PR&950DT products</term>: When `weight` is `torch.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, or `torch_npu.hifloat8`, `weight_dtype` must be set to `torch.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, or `torch_npu.hifloat8`, respectively.
  <!-- end id25 -->

- **`weight_scale_dtype`** (`int`): Optional. Actual data type of the input `weight_scale`. The default value is `None`, indicating that the actual data type of `weight_scale` is the same as its `dtype`.

  <!-- npu="A3,910b" id26 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, this parameter is not supported. Use the default value.
  <!-- end id26 -->
  <!-- npu="950" id27 -->
  - <term>Ascend 950PR&950DT products</term>: When `weight_scale` is `torch.float8_e8m0fnu`, `weight_scale_dtype` must be set to `torch_npu.float8_e8m0fnu`.
  <!-- end id27 -->

- **`x_scale_dtype`** (`int`): Optional. Actual data type of the input `x_scale`. The default value is `None`, indicating that the actual data type of `x_scale` is the same as its `dtype`.

  <!-- npu="A3,910b" id28 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: Currently, this parameter is not supported. Use the default value.
  <!-- end id28 -->
  <!-- npu="950" id29 -->
  - <term>Ascend 950PR&950DT products</term>: When `x_scale` is `torch.float8_e8m0fnu`, `x_scale_dtype` must be set to `torch_npu.float8_e8m0fnu`.
  <!-- end id29 -->

## Return Values

- **`output`** (`Tensor`): Quantized output, $Q$ in the formulas. The data layout can be `ND`. Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id30 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data type can be `torch.int8`. The shape must be 2D with shape `[m, n/2]`.
  <!-- end id30 -->
  <!-- npu="950" id31 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch.float4_e2m1fn_x2`, `torch.int8`, `torch_npu.hifloat8`, or `torch_npu.float4_e1m2fn_x2` (supported only when `weight` is in $FRACTAL_NZ$ format). The `shape` supports 2D `[m, n / 2]`.
  <!-- end id31 -->

- **`output_scale`** (`Tensor`): Quantization factor for the output, $Q_{\text{scale}}$ in the formulas. The data layout can be `ND`. Non-contiguous tensors are supported.

  <!-- npu="A3,910b" id32 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The data type can be `torch.float`, and the `shape` supports 1D `[m]`.
  <!-- end id32 -->
  <!-- npu="950" id33 -->
  - <term>Ascend 950PR&950DT products</term>:
    - MX quantization scenario: The data type can be `torch.float8_e8m0fnu`, and the shape supports 3 dimensions `[m, ceil((n / 2) / 64), 2]`.
    - Pertoken quantization scenario: The shape supports 1 dimension `[m]`, and the data type can be `torch.float32`.
  <!-- end id33 -->

## Constraints

- This API can be used in inference and training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- The first dimension of `group_list` supports a maximum of 1024, that is, a maximum of 1024 groups.

<!-- npu="950" id34 -->
- WeightNZ scenario description (applicable only to Ascend 950PR&950DT products):
  - MXFP4 and MXFP8 scenarios support static graph mode but not dynamic graph mode.
  - Single-multi-single MXFP4 and MXFP8 scenarios support single-operator mode but not graph mode.
  - In MX quantization scenarios where `weight` is in ND format, when the input is of the `torch.float8_e4m3fn` or `torch.float8_e5m2` data type, N must be aligned to 2; when the input is of the `torch.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2` data type, N must be aligned to 4.
  - In MX quantization scenarios where `weight` is in NZ format, when the input is of the `torch.float8_e4m3fn` or `torch.float8_e5m2` data type, N must be aligned to 64; when the input is of the `torch.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2` data type, N must be aligned to 128.
  - In MXFP4 scenarios, `k = 2` is not supported, and `k` must be even.
<!-- end id34 -->
- Variables used in tensor shapes in the parameter descriptions:
  - `e`: Number of groups, ranging from 1 to 1024.
  - `m`: Size of the second-to-last dimension of the output matrix `output`, ranging from 1 to 2147483647.
  - `k`: Size of the reduction axis for matrix multiplication, ranging from 1 to 2147483647.

    <!-- npu="950" id48 -->
    MX quantization scenario requirements (applicable only to Ascend 950PR&950DT products):

    - In MX quantization scenarios where `weight` is in ND format, when the input is of the `torch.float8_e4m3fn` or `torch.float8_e5m2` data type, `n` must be aligned to 2; when the input is of the `torch.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2` data type, `n` must be aligned to 4.
    - In MX quantization scenarios where `weight` is in NZ format, when the input is of the `torch.float8_e4m3fn` or `torch.float8_e5m2` data type, `n` must be aligned to 64; when the input is of the `torch.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2` data type, `n` must be aligned to 128.
    <!-- end id48 -->

  - k: size of the matrix multiplication reduce axis. The value range is 1-2147483647.

- The data type combinations supported by the input and output Tensors are as follows:

    <!-- npu="A3,910b" id44 -->
    **Table 1** Atlas A2 products, Atlas A3 products

    | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` | `torch.float32` | `torch.int32` | `torch.float32` | `torch.float32` | `torch.int8` | `torch.float32` |
    <!-- end id44 -->

    <!-- npu="950" id45 -->
    **Table 2** Ascend 950PR&950DT products

    | Quantization Mode | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | MXFP8 quantization (ND layout) | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e8m0fnu` |
    | MXFP4 quantization (ND format) | `torch.float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported currently | Not supported currently | Not supported currently | `torch.float4_e2m1fn_x2` | `torch.float8_e8m0fnu` |
     | MXFP4 quantization (ND layout) | `torch.float4_e2m1fn_x2` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e8m0fnu` |
    | MXFP8 quantization (FRACTAL_NZ layout) | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |
    | MXFP4 quantization (FRACTAL_NZ format) | `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | `torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported currently | Not supported currently | Not supported currently | `torch.float4_e2m1fn_x2`/`torch.float8_e4m3fn`/`torch_npu.float4_e1m2fn_x2` | `torch.float8_e8m0fnu` |
    | Pertoken quantization | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32`/`torch.float16`/`torch.bfloat16` | `torch.float32` | Not supported currently | Not supported currently | Not supported currently | `torch.int8` | `torch.float32` |
    | Pertoken quantization | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.float32`/`torch.bfloat16` | `torch.float32` | Not supported currently | Not supported currently | Not supported currently | `torch_npu.hifloat8` | `torch.float32` |
    | Pertoken quantization | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.int64` | `torch.float32`/`torch.bfloat16` | `torch.float32` | Not supported currently | Not supported currently | Not supported currently | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float32` |
    | MxFP8FP4 quantization (FRACTAL_NZ format) | `torch.float8_e4m3fn` | `torch.float4_e2m1fn_x2` | `torch.int64` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | Not supported currently | Not supported currently | Not supported currently | `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |

    > **MxA8W4 scenario**:
    > - The data type of `x` is `torch.float8_e4m3fn`, and the data type of `weight` is `torch.float4_e2m1fn_x2`. The data format of `weight` must be FRACTAL_NZ, and the ND to FRACTAL_NZ format conversion can be implemented through the `torch_npu.npu_format_cast` API. `k` must be aligned to 32, and N must be aligned to 128.
    > - The single-single-single and single-multi-single scenarios are supported.
    <!-- end id45 -->

- Depending on the different numbers of input `x`, input `weight`, and output `y` tensors, the following scenarios are supported. In the scenarios, single indicates a single tensor, and multi indicates multiple tensors. The scenario order is x, weight, y. For example, single-multi-single indicates that x is a single tensor, weight is multiple tensors, and y is a single tensor.

    | Supported Scenario | Scenario Description | Scenario Restrictions |
    | --- | --- | --- |
    | Single-multi-single | x is a single tensor, weight is multiple tensors, and y is a single tensor. | 1. `group_list` must be passed, and the last value must be equal to the first dimension of the tensor in x.<br>2. The tensors in x, weight, and y must be 2D.<br>3. The N axis of each tensor in weight must be equal.<br>4. `group_list` must be passed. When `group_list_type` is 0, the last value must be equal to the first dimension of the tensor in x. When `group_list_type` is 1, the sum of the values must correspond one-to-one to the first dimension of the tensor in x, and the maximum length is 128. |

<!-- npu="950" id46 -->
- The supported shape combinations of input and output Tensors are as follows:

    **Table 3** Ascend 950PR&950DT products

    | Supported Scenario | Quantization Mode | x | weight | weight_scale | x_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Single-single-single | MX quantization (ND format) | (m, k) | <li>The non-transposed shape is in the form of {(e, k, n)}.</li><li>The transposed shape is in the form of {(e, n, k)}.</li> | <li>The non-transposed shape is in the form of {(e, ceil(k / 64), n, 2)}.</li><li>The transposed shape is in the form of {(e, n, ceil(k / 64), 2)}.</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | Single-single-single | MXFP8 quantization (FRACTAL_NZ format) | (m, k) | <li>The non-transposed shape is in the form of {(e, n/32, k/16, 16, 32)}.</li><li>The transposed shape is in the form of {(e, k/32, n/16,16, 32)}.</li> | <li>The non-transposed shape is in the form of {(e, ceil(k / 64), n, 2)}.</li><li>The transposed shape is in the form of {(e, n, ceil(k / 64), 2)}.</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | Single-single-single | MXFP4 quantization (FRACTAL_NZ format) | (m, k) | <li>The non-transposed shape is in the form of {(e, n/64, k/16, 16, 64)}.</li><li>The transposed shape is in the form of {(e, k/64, n/16,16, 64)}.</li> | <li>The non-transposed shape is in the form of {(e, ceil(k / 64), n, 2)}.</li><li>The transposed shape is in the form of {(e, n, ceil(k / 64), 2)}.</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | Single-single-single | Pertoken quantization | (m, k) | <li>The non-transposed shape is in the form of {(e, k, n)}.</li><li>The transposed shape is in the form of {(e, n, k)}.</li> | The shape is in the form of {(e, n)}. | (m, ) | (m, n / 2) | (m, ) |
    | Single-single-single | MxFP8FP4 quantization (FRACTAL_NZ format) | (m, k) | The transposed shape is in the form of {(e, k/32, n/16,16, 32)} | The transposed shape is in the form of {(e, n, ceil(k / 64), 2)} | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | Single-multi-single | MXFP8 quantization (FRACTAL_NZ format) | (m, k) | <li>The non-transposed shape is in the form of {e values of (n/32, k/16, 16, 32)}.</li><li>The transposed shape is in the form of {e values of (k/32, n/16,16, 32)}.</li> | <li>The non-transposed shape is in the form of {e values of (ceil(k / 64), n, 2)}.</li><li>The transposed shape is in the form of {e values of (n, ceil(k / 64), 2)}.</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
    | Single-multi-single | MXFP4 quantization (FRACTAL_NZ format) | (m, k) | <li>The non-transposed shape is in the form of {e values of (n/64, k/16, 16, 64)}.</li><li>The transposed shape is in the form of {e values of (k/64, n/16,16, 64)}.</li> | <li>The non-transposed shape is in the form of {e values of (ceil(k / 64), n, 2)}.</li><li>The transposed shape is in the form of {e values of (n, ceil(k / 64), 2)}.</li> | (m, ceil(k / 64), 2) | (m, n / 2) | (m, ceil((n / 2) / 64), 2) |
<!-- end id46 -->

## Examples

- Single-operator call

  <!-- npu="A3,910b" id35 -->
  - Atlas A2 products, Atlas A3 products:

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weight_scale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weight_scale.npu()], xScale.npu(), groupList.npu())
    ```
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - Ascend 950PR&950DT products: mx quantization scenario example - mxfp8

    ```python
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch_npu
    import math

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)

        weight_scale = torch.randint(low=-128, high=127, size=(E, math.ceil(K / 64), N, 2), dtype=torch.int8)
        xScale = torch.randint(low=-128, high=127, size=(M, math.ceil(K / 64), 2), dtype=torch.int8)
        groupList = torch.tensor([int(M/2), int(M/2)], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    K = 2
    E = 2
    M = 16
    N = 128
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = weight.npu()
    weight_scale = weight_scale.npu()
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(),
    [weight_npu], [weight_scale],
    xScale.npu(), groupList.npu(),
    dequant_mode = 2,
    quant_mode=2,
    dequant_dtype=torch.float32,
    quant_dtype=torch.float8_e4m3fn,
    weight_scale_dtype=torch_npu.float8_e8m0fnu,
    x_scale_dtype=torch_npu.float8_e8m0fnu)
    ```
  <!-- end id36 -->
  <!-- npu="950" id37 -->
  - Ascend 950PR&950DT products: mx quantization scenario example - mxfp4

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math

    K = 9
    E = 2
    M = 2255
    N = 896

    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).npu()
    weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8).npu()
    weight_scale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8).npu()
    xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()

    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weight_scale],
        xScale, groupList,
        dequant_mode=2,
        dequant_dtype=torch.float32,
        quant_mode=2,
        quant_dtype=torch_npu.float4_e2m1fn_x2,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
        x_dtype=torch_npu.float4_e2m1fn_x2,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        group_list_type=1)

    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```
  <!-- end id37 -->
  <!-- npu="950" id38 -->
  - Ascend 950PR&950DT products: pertoken quantization scenario example

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math
    K = 9
    E = 2
    M = 2255
    N = 896
    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weight = torch.randint(0, 256, (E, K, N), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weight_scale = torch.randint(0, 256, (E, N), dtype=torch.float).npu()
    xScale = torch.randint(0, 256, (M,), dtype=torch.float).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()
    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weight_scale],
        xScale, groupList,
        dequant_mode=0,
        quant_mode=0,
        quant_dtype=torch_npu.float8_e5m2,
        dequant_dtype=torch.float,
        group_list_type=1)
    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```
  <!-- end id38 -->
  <!-- npu="950" id39 -->
  - Ascend 950PR&950DT products: MxA8W4 fake-quantization scenario example

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    from ml_dtypes import float8_e4m3fn

    def ceil_div(a, b):
        return math.ceil(a / b)
    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, x_scale, weight_scale, group_list, group_list_type, dequant_mode, dequant_dtype,
                    quant_mode, quant_dtype):
            weight = weight.transpose(-1, -2)
            weight_scale = weight_scale.transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, [weight],
                                                                    weight_scale=[weight_scale],
                                                                    bias=None,
                                                                    x_scale=x_scale,
                                                                    dequant_mode=dequant_mode,
                                                                    dequant_dtype=dequant_dtype,
                                                                    quant_mode=quant_mode,
                                                                    quant_dtype=quant_dtype,
                                                                    group_list_type=group_list_type,
                                                                    group_list=group_list,
                                                                    weight_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    x_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output

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

    def generate_data_mxa8w4(m, n, k, group_num, x_range, weight_range, weight_scale_range, x_scale_range, group_size):
        x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
        x = x.to(torch.float8_e4m3fn)
        weight = torch.rand((group_num, n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
        weight = fp32_to_fp4_e2m1_u8packed(weight)
        pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                x_scale_range[1] - x_scale_range[0]) + x_scale_range[0]
        pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        scale = torch.rand((group_num, n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                weight_scale_range[1] - weight_scale_range[0]) + weight_scale_range[0]
        scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        return x, weight, pertoken_scale, scale

    def main():
        g, m, k, n, is_dynamic = 4, 128, 32, 512, True

        groupType = 0
        group_list_type = 1  # 0: cumsun 1: count
        dequant_mode = 2  # mx quantization
        dequant_dtype = torch.float32
        quant_mode = 2  # mx quantization
        quant_dtype = torch.float8_e4m3fn
        group_size = 32

        # generate data range
        x_range = [-1, 1]
        weight_range = [-6, 6]
        weight_scale_range = [0, 2]
        x_scale_range = [0, 2]
        x, weight, x_scale, weight_scale = generate_data_mxa8w4(m, n, k, g, x_range=x_range, weight_range=weight_range,
                                                                weight_scale_range=weight_scale_range,
                                                                x_scale_range=x_scale_range,
                                                                group_size=group_size)
        group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
        # npu
        x_npu = x.npu()
        weight_npu = weight.npu()
        x_scale_npu = x_scale.npu()
        weight_scale_npu = weight_scale.npu()
        group_list_npu = group_list.npu()
        # npu_format_cast
        weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch_npu.float4_e2m1fn_x2)
        model = NetPTA().npu()
        output, output_scale = model(x_npu, weight_npu, x_scale_npu, weight_scale_npu, group_list_npu, group_list_type,
                                        dequant_mode, dequant_dtype, quant_mode, quant_dtype)
        print("output")
        print(output)
    if __name__ == '__main__':
        main()
    ```
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - Ascend 950PR&950DT products: MxA8W4 fake-quantization single-multi-single scenario example

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    from ml_dtypes import float8_e4m3fn
    def ceil_div(a, b):
        return math.ceil(a / b)
    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, x_scale, weight_scale, group_list, group_list_type, dequant_mode, dequant_dtype,
                    quant_mode, quant_dtype):
            for i in range(len(weight)):
                weight[i] = weight[i].transpose(-1, -2)
                weight_scale[i] = weight_scale[i].transpose(-2, -3)
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight,
                                                                    weight_scale=weight_scale,
                                                                    bias=None,
                                                                    x_scale=x_scale,
                                                                    dequant_mode=dequant_mode,
                                                                    dequant_dtype=dequant_dtype,
                                                                    quant_mode=quant_mode,
                                                                    quant_dtype=quant_dtype,
                                                                    group_list_type=group_list_type,
                                                                    group_list=group_list,
                                                                    weight_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    x_scale_dtype=torch_npu.float8_e8m0fnu,
                                                                    weight_dtype=torch_npu.float4_e2m1fn_x2)
            return output
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
    def generate_data_mxa8w4(m, n, k, group_num, x_range, weight_range, weight_scale_range, x_scale_range, group_size):
        x = torch.rand((m, k), dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]
        x = x.to(torch.float8_e4m3fn)

        pertoken_scale = torch.rand((m, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                x_scale_range[1] - x_scale_range[0]) + x_scale_range[0]
        pertoken_scale = pertoken_scale.to(torch.float8_e8m0fnu).view(torch.uint8)
        weight_list = []
        weight_scale_list = []
        for i in range(group_num):
            weight = torch.rand((n, k), dtype=torch.float32) * (weight_range[1] - weight_range[0]) + weight_range[0]
            weight = fp32_to_fp4_e2m1_u8packed(weight)
            scale = torch.rand((n, ceil_div(k, group_size * 2), 2), dtype=torch.float32) * (
                    weight_scale_range[1] - weight_scale_range[0]) + weight_scale_range[0]
            scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
            weight_list.append(weight.npu())
            weight_scale_list.append(scale.npu())
        return x, weight_list, pertoken_scale, weight_scale_list
    def main():
        g, m, k, n, is_dynamic = 4, 128, 32, 512, True
        groupType = 0
        group_list_type = 1  # 0: cumsun 1: count
        dequant_mode = 2  # mx quantization
        dequant_dtype = torch.float32
        quant_mode = 2  # mx quantization
        quant_dtype = torch.float8_e4m3fn
        group_size = 32
        # generate data range
        x_range = [-1, 1]
        weight_range = [-6, 6]
        weight_scale_range = [0, 2]
        x_scale_range = [0, 2]
        x, weight, x_scale, weight_scale = generate_data_mxa8w4(m, n, k, g, x_range=x_range, weight_range=weight_range,
                                                                weight_scale_range=weight_scale_range,
                                                                x_scale_range=x_scale_range,
                                                                group_size=group_size)
        group_list = torch.Tensor([32, 32, 32, 32]).to(torch.int64)
        # npu
        x_npu = x.npu()
        x_scale_npu = x_scale.npu()
        group_list_npu = group_list.npu()
        # npu_format_cast
        for idx in range(g):
            weight[idx] = torch_npu.npu_format_cast(weight[idx], 29, customize_dtype=torch.float8_e4m3fn,
                                                    input_dtype=torch_npu.float4_e2m1fn_x2)
        model = NetPTA().npu()
        output, output_scale = model(x_npu, weight, x_scale_npu, weight_scale, group_list_npu, group_list_type,
                                        dequant_mode, dequant_dtype, quant_mode, quant_dtype)
        print("output")
        print(output)
    if __name__ == '__main__':
        main()
    ```
  <!-- end id40 -->

- Graph mode call
  <!-- npu="A3,910b" id47 -->
  - Atlas A2 products/Atlas A3 products

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
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weight_scale, xscale, group_list):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, xscale, group_list)
            return output

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weight_scale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weight_scale.npu()], xScale.npu(), groupList.npu())
    ```
  <!-- end id47 -->

  <!-- npu="950" id41 -->
  - Ascend 950PR&950DT products: mx quantization scenario example - mxfp8

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()

    npu_backend = tng.get_npu_backend(compiler_config=config)

    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weight_scale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)

        def forward(self, x_npu: torch.Tensor, w: torch.Tensor, group_list_npu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weight_scale = self.weight_scale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weight_scale.transpose(1, 2)], xScale.npu(), group_list_npu, quant_mode=2, quant_dtype=torch.float8_e5m2, dequant_mode=2, dequant_dtype=torch.float32,weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu)
                return y, y_scale

    def gen_input_data(E, M, K, N, transpose):
        if transpose:
            x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight_scale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8)
            xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
            groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList

    def run_npu(x, weight_npu, weight_scale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weight_scale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        for k in range(1):
            torch_npu.npu.synchronize()
            custom_output, y_scale = model(x, None, groupList)
            torch_npu.npu.synchronize()

    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = True
        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weight_scale_npu, xScale_npu, groupList_npu, transpose)
    ```
  <!-- end id41 -->
  <!-- npu="950" id42 -->
  - Ascend 950PR&950DT products: mx quantization scenario example - mxfp4

    ```python
    import os
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,
                    x,
                    weight,
                    weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode,
                    dequant_dtype,
                    quant_mode,
                    quant_dtype,
                    group_list_type,
                    weight_scale_dtype,
                    x_scale_dtype,
                    x_dtype=None,
                    weight_dtype=None,
                    transpose_w=False):
            if quant_dtype is None:
                quant_dtype = torch.float8_e5m2
            if weight_scale_dtype is None:
                weight_scale_dtype = torch_npu.float8_e8m0fnu
            if x_scale_dtype is None:
                x_scale_dtype = torch_npu.float8_e8m0fnu

            processed_weight = []
            for w in weight:
                if transpose_w:
                    w = w.transpose(1, 2)
                processed_weight.append(w)

            processed_weight_scale = []
            for ws in weight_scale:
                if transpose_w:
                    ws = ws.transpose(1, 2)
                processed_weight_scale.append(ws)

            with torch.no_grad():
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x,
                    processed_weight,
                    processed_weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode=dequant_mode,
                    dequant_dtype=dequant_dtype,
                    quant_mode=quant_mode,
                    quant_dtype=quant_dtype,
                    group_list_type=group_list_type,
                    weight_scale_dtype=weight_scale_dtype,
                    x_scale_dtype=x_scale_dtype,
                    x_dtype=x_dtype,
                    weight_dtype=weight_dtype
                )
                return y, y_scale

    def gen_input_data(E, M, K, N):
        x = torch.randint(0, 256, (M, K), dtype=torch.uint8)
        weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8)
        weight_scale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8)
        xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList

    if __name__ == "__main__":
        K = 9
        E = 2
        M = 2255
        N = 896
        transpose = False

        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        weight_list = [weight_npu]
        weight_scale_list = [weight_scale_npu]

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

        y, y_scale = model(
            x_npu,
            weight_list,
            weight_scale_list,
            xScale_npu,
            groupList_npu,
            dequant_mode=2,
            dequant_dtype=torch.float32,
            quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            group_list_type=1,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            transpose_w=transpose
        )

        print("y shape: ", y.shape)
        print("y_scale shape: ", y_scale.shape)
    ```
  <!-- end id42 -->
  <!-- npu="950" id43 -->
  - Ascend 950PR&950DT products: pertoken quantization scenario example

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"
    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weight_scale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)
        def forward(self, x_npu: torch.Tensor, w: torch.Tensor, group_list_npu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weight_scale = self.weight_scale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weight_scale], xScale.npu(), group_list_npu, quant_mode=0, quant_dtype=torch.float8_e5m2, dequant_mode=0, dequant_dtype=torch.float)
                return y, y_scale
    def gen_input_data(E, M, K, N, transpose):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight_scale = torch.randint(low=0, high=256, size=(E, N), dtype=torch.float)
        xScale = torch.randint(low=0, high=256, size=(M,), dtype=torch.float)
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weight_scale, xScale, groupList
    def run_npu(x, weight_npu, weight_scale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weight_scale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        for k in range(1):
            torch_npu.npu.synchronize()
            customyy_output, y_scale = model(x, None, groupList)
            print(customyy_output, y_scale)
            torch_npu.npu.synchronize()
    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = False
        x, weight, weight_scale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weight_scale_npu = weight_scale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weight_scale_npu, xScale_npu, groupList_npu, transpose)
    ```
  <!-- end id43 -->
