# torch_npu.npu_grouped_matmul_swiglu_quant_v2

> [!NOTICE]  
> This API is updated in this version. For details about the specific changes, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

- Provides an efficient method to perform fused computation of grouped matrix multiplication (`GroupedMatMul`), dequantization (`dequant`), the `SwiGLU` activation function, and quantization (`quant`). This method is applicable to scenarios where the output of matrix multiplication requires `SwiGLU` activation. The fused operator enables partial parallel execution at the kernel level, improving computational efficiency. The following quantization scenarios are supported.

- Formulas
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    <details>
    <summary>A8W8 quantization scenarios (A indicates the activation matrix, W indicates the weight matrix, and 8 indicates the int8 data type):</summary>

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

           $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

           $C_{i,act}, gate_{i} = split(C_{i})$

           $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$
        3. Quantize the output results.

           $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

           $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

    <details>
    <summary>A8W4 quantization scenarios (MSD) (A indicates the activation matrix, W indicates the weight matrix, 8 indicates the int8 data type, and 4 indicates the int4 data type):</summary>

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
        2. Split the `int8` left matrix input into high and low 4-bit parts.

          $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$，$X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8$
        3. Perform matrix multiplication separately for the high and low parts, apply `perchannel` or `pergroup` quantization scaling, and combine the results with the auxiliary matrix.

          $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$
        4. Quantize the output results.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

    <details>
    <summary>A4W4 quantization scenarios (A indicates the activation matrix, W indicates the weight matrix, and 4 indicates the int4 data type):</summary>

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

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

          $S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}$

          Note: When the shape of `smoothScale` is `(E,)`, it is broadcasted to match the shape of $S_{i}$.
        3. Quantize the output results.

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>
  
  - <term>Ascend 950PR/Ascend 950DT</term>：  
    <details>
    <summary>MX quantization scenarios:</summary>

      1. Perform the following computation using the inputs determined based on the grouping:

         $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

         $C_{i,act}, gate_{i} = split(C_{i})$

         $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

      2. Quantize the output:

         $shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax$

         $Q\_scale = 2 ^ {shared\_exp}$

         $Q_i = quantize\_to\_element\_format(S_i/Q\_scale), \space i\space from\space 1\space to\space blocksize$

         where $emax$ denotes the exponent of the maximum normal value for the corresponding data type:

         |   DataType    | emax |
         | :-----------: | :--: |
         | FLOAT8_E4M3FN |  8   |
         |  FLOAT8_E5M2  |  15  |
         |  FLOAT4_E2M1  |  2   |

         where $blocksize$ denotes the number of elements per quantization block. Only `32` is supported.

    </details>

    <details>
    <summary>pertoken quantization scenarios:</summary>

      1. Perform the following computation using the inputs determined based on the grouping:：

         $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i} \odot w\_scale_{i}$

         $C_{i,act}, gate_{i} = split(C_{i})$

         $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

         where $x\_scale_{i}$ denotes the quantization factor for the corresponding token.

      2. Quantize the output:

         $Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}$

         $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

## Prototype

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Left matrix for matrix multiplication, $X$ in the formulas. This parameter must be 2D with shape `[m, k]`. The data layout can be ND. Non-contiguous tensors are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `int4`, `int8`, or `int32`.
  - <term>Ascend 950PR/Ascend 950DT</term>: The data type can be `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, `torch.int8`, or `torch_npu.hifloat8`. For `torch_npu.hifloat8` and the `float4` series, the optional parameter `x_dtype` must be set to the corresponding data type. In this case, the data type of `x` itself is ignored, but `x` must still have an 8-bit data type to ensure the shape is correct. For the `float4` series, the inner dimension `k` must be even so that 8 bits can be converted into two `float4` values. The data layout is `ND`.

- **`weight`** (`TensorList`): Required. Weight matrix (the right matrix for matrix multiplication), $W$ in the formulas. Currently, only a `TensorList` of length `1` is supported. This parameter must be 3D with shape `[e, k, n]` (in ND layout), or a 5D tensor in NZ layout. The data layout can be ND or FRACTAL_NZ, which can be converted using `npu_format_cast`. Non-contiguous tensors are supported. 
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
    - The data type can be `int4`, `int8`, or `int32`. `int32` is used for adaptation in A8W4 and A4W4 scenarios. In practice, a single `int32` value is interpreted as eight `int4` elements.
    - In A8W8 scenarios, `weight` supports only the `FRACTAL_NZ` layout and does not support the `ND` layout.
  - <term>Ascend 950PR/Ascend 950DT</term>:
    - When the data layout is `ND`, 3D shapes are supported: non-transposed shape `(e, k, n)` and transposed shape `(e, n, k)`. Supported data types are `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, `torch.int8`, and `torch_npu.hifloat8`. For `torch_npu.hifloat8` and the `float4` series, the optional parameter `weight_dtype` must be set to the corresponding data type. In this case, the data type of `weight` itself is ignored, but `weight` must still have an 8-bit data type to ensure the shape is correct. For the `float4` series, the inner dimension must be even so that 8 bits can be converted into two `float4` values.
    - When the data layout is `FRACTAL_NZ` (conversion can be performed using `npu_format_cast`), 5D shapes are supported: non-transposed shape `(e, n/32, k/16, 16, 32)` and transposed shape `(e, k/32, n/16, 16, 32)`. Only `torch.float8_e4m3fn` is supported.

- **`weight_scale`** (`TensorList`): Required. Quantization factor for the weight matrix,  $w_{scale}$ in the formulas. Currently, only a `TensorList` of length `1` is supported. The data layout can be ND. Non-contiguous tensors are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: When the data type of `weight` is `int8`, the shape of `weight_scale` can have 2 dimensions. When the data type of `weight` is `int32`, the shape of `weight_scale` can have 2 or 3 dimensions. The data type can be `float32`, `float16`, `bfloat16`, or `uint64`.
  - <term>Ascend 950PR/Ascend 950DT</term>: In MX quantization scenarios, this parameter can be 4D with shape `(e, ceil(k / 64), n, 2)` (non-transposed) or `(e, n, ceil(k / 64), 2)` (transposed); and the data type can be `torch_npu.float8_e8m0fnu`. In `pertoken` quantization scenarios, this parameter can be 2D with shape `(e, n)`. When `x` is `torch.int8`, the data type of `weight_scale` can be `torch.bfloat16`, `torch.float32`, or `torch.float16`. When `x` is `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`, the data type of `weight_scale` can be `torch.bfloat16` or `torch.float32`.

- **`x_scale`** (`Tensor`): Required. Quantization factor for the activation matrix, $x\_scale$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: This parameter must be 1D with shape `(m)`. The data type can be `float32`.
  - <term>Ascend 950PR/Ascend 950DT</term>: In MX quantization scenarios, this parameter can be 3D with shape `(m, ceil(k / 64), 2)`. The data type can be `torch_npu.float8_e8m0fnu`. In `pertoken` quantization scenarios, this parameter must be 1D with shape `(m)`. The data type can be `torch.float32`.

- **`group_list`** (`Tensor`): Required. Number of tokens in each group involved in the computation, $groupList$ in the formulas. This parameter must be 1D with shape `[e]`, and its length must be identical to the first axis dimension of `weight`. The data type can be `int64`. The data layout can be ND. Non-contiguous tensors are supported.
- **`smooth_scale`** (`Tensor`): Optional. Smooth scaling factor, $smoothScale$ in the formulas. The data type can be `float32`. The data layout can be ND. This parameter must be provided only in A4W4 scenarios, and its first axis length must be identical to the first axis dimension of `weight`. The shape of this parameter is `(E, N/2)` or `(E,)`. When shape `(E,)` is used, broadcast multiplication is applied. In other scenarios, the default value is `None`.
- **`weight_assist_matrix`** (`TensorList`): Optional. Auxiliary matrix for the right matrix, $weightAssistMatrix$ in the formulas. The data type can be `float32`. The data layout can be ND. This parameter must be a 2D tensor. This parameter must be provided only in A8W4 scenarios, where the length of its first dimension must be identical to that of the first dimension of `weight`, and the length of its last dimension must be identical to that of the last dimension of `weight` when restored to the ND layout. In other scenarios, the default value is `None`.
- **`bias`** (`Tensor`): Optional. Offset value for matrix multiplication computation, $bias$ in the formulas. This parameter must be a 2D tensor. The data type can be `int32`. Currently, only the default value `None` is supported.
- **`dequant_mode`** (`int`): Optional. Dequantization mode. This parameter is of type `int32` and has a default value of `0`. A value of `0` indicates `pertoken` quantization for the activation matrix and `perchannel` quantization for the weight matrix. A value of `1` indicates `pertoken` quantization for the activation matrix and `pergroup` quantization for the weight matrix. A value of `2` indicates MX quantization.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: In A8W4 scenarios, `dequant_mode` can be `0` or `1`. In A8W8 and A4W4 scenarios, `dequant_mode` can only be `0`.
  - <term>Ascend 950PR/Ascend 950DT</term>: Currently, only `0` and `2` are supported.

- **`dequant_dtype`** (`int`): Optional. Dequantization data type. This parameter is of type `int32`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, only the default value `0` (indicating `float32`) is supported.
  - <term>Ascend 950PR/Ascend 950DT</term>: The default value is `torch.int8`. Currently, `torch.float32`, `torch.bfloat16`, and `torch.float16` are supported.

- **`quant_mode`** (`int`): Optional. Quantization mode after SwiGLU. This parameter is of type `int32`. Valid values are `0` (default, `pertoken` quantization), `1` (`pergroup` quantization), or `2` (MX quantization).
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, only the default value `0` (`pertoken` quantization) is supported.
  - <term>Ascend 950PR/Ascend 950DT</term>: Currently, only `0` and `2` are supported.

- **`quant_dtype`** (`int`): Optional. Low-bit data type after quantization. This parameter is of type `int32`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, only the default value `0` (indicating `int8`) is supported.
  - <term>Ascend 950PR/Ascend 950DT</term>: The default value is `torch.int8`. Currently, `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, `torch.int8`, and `torch_npu.hifloat8` are supported.

- **`group_list_type`** (`int`): Optional. Input type of `group_list`. This parameter is of type `int32` and has a default value of `0`.
  - A value of `0` indicates cumsum mode, where each element in `group_list` represents the cumulative length of the current group.
  - A value of `1` indicates count mode, where each element in `group_list` represents the number of elements in the corresponding group.
- **`tuning_config`** (`List[int]`): Optional. The first element in this parameter array specifies the expected number of tokens processed by each expert. Elements from the second element onward are reserved for future expansion and do not need to be specified by the user. The default value is `None`.

- **`x_dtype`** (`int`): Optional. Actual data type of the input `x`. Currently, only the default value `None` is supported, indicating that the actual data type of `x` is the same as its `dtype`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, this parameter is not supported. Use the default value.
  - <term>Ascend 950PR/Ascend 950DT</term>: When `x` is `float4_e2m1fn_x2` or `hifloat8`, `x_dtype` must be set to `torch_npu.float4_e2m1fn_x2` or `torch_npu.hifloat8`, respectively.

- **`weight_dtype`** (`int`): Optional. Actual data type of the input `weight`. Currently, only the default value `None` is supported, indicating that the actual data type of `weight` is the same as its `dtype`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, this parameter is not supported. Use the default value.
  - <term>Ascend 950PR/Ascend 950DT</term>: When `weight` is `float4_e2m1fn_x2` or `hifloat8`, `weight_dtype` must be set to `torch_npu.float4_e2m1fn_x2` or `torch_npu.hifloat8`, respectively.

- **`weight_scale_dtype`** (`int`): Optional. Actual data type of the input `weight_scale`. The default value is `None`, indicating that the actual data type of `weight_scale` is the same as its `dtype`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, this parameter is not supported. Use the default value.
  - <term>Ascend 950PR/Ascend 950DT</term>: When `weight_scale` is `float8_e8m0fnu`, `weight_scale_dtype` must be set to `torch_npu.float8_e8m0fnu`.

- **`x_scale_dtype`** (`int`): Optional. Actual data type of the input `x_scale`. The default value is `None`, indicating that the actual data type of `x_scale` is the same as its `dtype`.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: Currently, this parameter is not supported. Use the default value.
  - <term>Ascend 950PR/Ascend 950DT</term>: When `x_scale` is `float8_e8m0fnu`, `x_scale_dtype` must be set to `torch_npu.float8_e8m0fnu`.

## Return Values

- **`output`** (`Tensor`): Quantized output, $Q$ in the formulas. The data layout can be `ND`. Non-contiguous tensors are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `int8`. The shape must be 2D with shape `[m, n/2]`.
  - <term>Ascend 950PR/Ascend 950DT</term>: The data type can be `torch.float8_e4m3fn`, `torch.float8_e5m2`, `torch_npu.float4_e2m1fn_x2`, `torch.int8`, or `torch_npu.hifloat8`. The shape can be 2D with shape `[m, n/2]`.

- **`output_scale`** (`Tensor`): Quantization factor for the output, $Q_{\text{scale}}$ in the formulas. The data layout can be `ND`. Non-contiguous tensors are supported.
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float32`. The shape must be 1D with shape `(m)`.
  - <term>Ascend 950PR/Ascend 950DT</term>: In MX quantization scenarios, the data type can be `torch_npu.float8_e8m0fnu`, and the parameter can be 3D with shape `(m, ceil((n/2)/64), 2)`. In `pertoken` quantization scenarios, the shape must be 1D with shape `(m)`, and the data type can be `torch.float32`.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- Deterministic computation: This API defaults to a deterministic implementation. For identical inputs, multiple execution passes generate identical outputs to guarantee repeatability.
- In MX quantization scenarios, `n` must be aligned to 128.
- In MXFP4 scenarios, `k = 2` is not supported, and `k` must be even.
- The first dimension of `group_list` supports a maximum of 1024, meaning that up to 1024 groups are supported.
- Variables used in tensor shapes in the parameter descriptions:
  - `e`: Number of groups, ranging from 1 to 1024.
  - `m`: Size of the second-to-last dimension of the output matrix `output`, ranging from 1 to 2147483647.
  - `n`: Twice the size of the last dimension of the output matrix `output`, ranging from 1 to 2147483647. In MX quantization scenarios on <term>Ascend 950PR/Ascend 950DT</term>, `n` must be aligned to 128.
  - `k`: Size of the reduction axis for matrix multiplication, ranging from 1 to 2147483647.

- Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products:
    - The A8W8, A8W4 and A4W4 quantization scenarios are supported. The following table describes the data type configurations supported by the input and output tensors.

        |Quantization Scenario|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|`int8`|`int8`|`float32`, `float16`, `bfloat16`|`float32`|-|`int8`|`float32`|
        |A8W4|`int8`|`int4`, `int32`|`uint64`|`float32`|-|`int8`|`float32`|
        |A4W4|`int4`, `int32`|`int4`, `int32`|`float32`|`float32`|`float32`|`int8`|`float32`|

    - The following table describes the shape constraints.

        |Quantization Scenario|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|(M, K)|The shape in NZ format is `{(E, N/32, K/16, 16, 32)}`.|{(E, N)}|(M,)|-|(M, N/2)|(M,)|
        |A8W4|(M, K)|The shape in ND layout `{(E, K, N)}` or the shape in NZ layout.|`perchannel`:{(E, N)}; `pergroup`:{(E, K\_group\_num, N)}|(M,)|-|(M, N/2)|(M,)|
        |A4W4|(M, K)|The shape in ND layout `{(E, K, N)}` or the shape in NZ layout.|{(E, N)}|(M,)|`(E, N/2)` or `(E,)`|(M, N/2)|(M,)|

    - In A8W8 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 65536.
    - In A8W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.
    - In A4W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.
- <term>Ascend 950PR/Ascend 950DT</term>:
  - Supported data type combinations for input and output tensors are as follows:

    - MX quantization scenarios:

      | Quantization Mode | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | MXFP8 quantization (ND layout) | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` |
      | MXFP4 quantization (ND layout) | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch_npu.float4_e2m1fn_x2` / `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` |
      | MXFP8 quantization (FRACTAL_NZ layout) | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` |

    - `pertoken` quantization scenarios:

      | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.int64` | `torch.float32` / `torch.bfloat16` | `torch.float32` | Not supported | Not supported | Not supported | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float32` |
      | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` / `torch.float16` / `torch.bfloat16` | `torch.float32` | Not supported | Not supported | Not supported | `torch.int8` | `torch.float32` |
      | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.float32` / `torch.bfloat16` | `torch.float32` | Not supported | Not supported | Not supported | `torch_npu.hifloat8` | `torch.float32` |

  - Supported shape combinations for input and output tensors are as follows:

    | Quantization Mode | x | weight | weight_scale | x_scale | output | output_scale |
    | --- | --- | --- | --- | --- | --- | --- |
    | MX quantization (ND layout) | `(m, k)` | Non-transposed shape: `(e, k, n)`<br>Transposed shape: `(e, n, k)` | Non-transposed shape: `(e, ceil(k / 64), n, 2)`<br>Transposed shape: `(e, n, ceil(k / 64), 2)` | `(m, ceil(k / 64), 2)` | `(m, n / 2)` | `(m, ceil((n / 2) / 64), 2)` |
    | MX quantization (FRACTAL_NZ layout) | `(m, k)` | Non-transposed shape: `(e, n / 32, k / 16, 16, 32)`<br>Transposed shape: `(e, k / 32, n / 16, 16, 32)` | Non-transposed shape: `(e, ceil(k / 64), n, 2)`<br>Transposed shape: `(e, n, ceil(k / 64), 2)` | `(m, ceil(k / 64), 2)` | `(m, n / 2)` | `(m, ceil((n / 2) / 64), 2)` |
    | `pertoken` quantization | `(m, k)` | Non-transposed shape: `(e, k, n)`<br>Transposed shape: `(e, n, k)` | `(e, n)` | `(m,)` | `(m, n / 2)` | `(m,)` |

## Examples

- Single-operator call
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax
    
    torch.npu.config.allow_internal_format = True
    
    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList    
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu())
    ```
  
  - <term>Ascend 950PR/Ascend 950DT</term>: MX quantization example (mxfp8)

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

        weightScale = torch.randint(low=-128, high=127, size=(E, math.ceil(K / 64), N, 2), dtype=torch.int8)
        xScale = torch.randint(low=-128, high=127, size=(M, math.ceil(K / 64), 2), dtype=torch.int8)
        groupList = torch.tensor([int(M / 2), int(M / 2)], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    K = 2
    E = 2
    M = 16
    N = 128
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = weight.npu()
    weightScale = weightScale.npu()
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x.npu(),
        [weight_npu],
        [weightScale],
        xScale.npu(),
        groupList.npu(),
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu)
    ```

  - <term>Ascend 950PR/Ascend 950DT</term>: MX quantization example (mxfp4)

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
    weightScale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8).npu()
    xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()

    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x,
        [weight], [weightScale],
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

  - <term>Ascend 950PR/Ascend 950DT</term>: `pertoken` quantization example

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
    weightScale = torch.randint(0, 256, (E, N), dtype=torch.float).npu()
    xScale = torch.randint(0, 256, (M,), dtype=torch.float).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()
    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weightScale],
        xScale, groupList,
        dequant_mode=0,
        quant_mode=0,
        quant_dtype=torch_npu.float8_e5m2,
        dequant_dtype=torch.float,
        group_list_type=1)
    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```

- Graph mode call
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
    
    torch.npu.config.allow_internal_format = True
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weightscale, xscale, group_list, quant_dtype):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weightscale, xscale, group_list, quant_dtype=quant_dtype, dequant_dtype=5)
            return output    
     
    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList    
    E = 2
    M = 512
    K = 7168
    N = 4096
    quant_dtype = 1
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu(), quant_dtype)
    ```

  - <term>Ascend 950PR/Ascend 950DT</term>: MX quantization example (mxfp8)

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
        def __init__(self, weight_npu, weightScale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weightScale = nn.Parameter(weightScale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)

        def forward(self, x_npu: Torch.Tensor, w: Torch.Tensor, group_list_npu: Torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weightScale = self.weightScale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weightScale.transpose(1, 2)], xScale.npu(), group_list_npu, quant_mode=2, quant_dtype=torch.float8_e5m2, dequant_mode=2, dequant_dtype=torch.float32,weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu)
                return y, y_scale

    def gen_input_data(E, M, K, N, transpose):
        if transpose:
            x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weightScale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8)
            xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
            groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    def run_npu(x, weight_npu, weightScale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weightScale, xScale, transpose).npu()
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
        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weightScale_npu, xScale_npu, groupList_npu, transpose)
    ```

  - <term>Ascend 950PR/Ascend 950DT</term>: MX quantization example (mxfp4)

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
        weightScale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8)
        xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    if __name__ == "__main__":
        K = 9
        E = 2
        M = 2255
        N = 896
        transpose = False

        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        weight_list = [weight_npu]
        weight_scale_list = [weightScale_npu]

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

  - <term>Ascend 950PR/Ascend 950DT</term>: `pertoken` quantization example

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
        def __init__(self, weight_npu, weightScale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weightScale = nn.Parameter(weightScale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)
        def forward(self, x_npu: Torch.Tensor, w: Torch.Tensor, group_list_npu: Torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weightScale = self.weightScale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weightScale], xScale.npu(), group_list_npu, quant_mode=0, quant_dtype=torch.float8_e5m2, dequant_mode=0, dequant_dtype=torch.float)
                return y, y_scale
    def gen_input_data(E, M, K, N, transpose):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weightScale = torch.randint(low=0, high=256, size=(E, N), dtype=torch.float)
        xScale = torch.randint(low=0, high=256, size=(M,), dtype=torch.float)
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList
    def run_npu(x, weight_npu, weightScale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weightScale, xScale, transpose).npu()
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
        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weightScale_npu, xScale_npu, groupList_npu, transpose)
    ```
