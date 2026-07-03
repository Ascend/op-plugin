# torch_npu.npu_grouped_matmul_swiglu_quant_v2

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

Provides an efficient method to perform fused computation of grouped matrix multiplication (`GroupedMatMul`), dequantization (`dequant`), the `SwiGLU` activation function, and quantization (`quant`). This method is applicable to scenarios where the output of matrix multiplication requires `SwiGLU` activation. The fused operator enables partial parallel execution at the kernel level, improving computational efficiency. The following quantization scenarios are supported.

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

       $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$, $X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8$

    3. Perform matrix multiplication separately for the high and low parts, apply `perchannel` or `pergroup` quantization scaling, and combine the results with the auxiliary matrix.

       $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

       $C_{i,act}, gate_{i} = split(C_{i})$

       $S_{i}=Swish(C_{i,act})\odot gate_{i}$, where $Swish(x)=\frac{x}{1+e^{-x}}$

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

## Prototype

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Left matrix for matrix multiplication, $X$ in the formulas. This parameter must be 2D with shape `[m, k]`. The data type can be `int8` or `int32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`weight`** (`TensorList`): Required. Weight matrix representing the right matrix in multiplication, $W$ in the formulas. Currently, only a `TensorList` of length `1` is supported. This parameter must be 3D with shape `[e, k, n]` (in ND layout), or a 5D tensor in NZ layout. The data type can be `int8`. The data layout can be ND or FRACTAL_NZ, which can be converted using `npu_format_cast`. Non-contiguous tensors are supported. In A8W8 quantization scenarios, only the NZ layout is supported, and the ND data layout is not allowed. The `int32` data type is used for adaptation in A8W4 and A4W4 scenarios, where a single `int32` value is interpreted as eight `int4` values.
- **`weight_scale`** (`TensorList`): Required. Quantization factor for the weight matrix,  $w\_scale$ in the formulas. Currently, only a `TensorList` of length `1` is supported. When the data type of `weight` is `int8`, this parameter is 2D. When the data type of `weight` is `int32`, this parameter can be 2D or 3D. The data type can be `float32`, `float16`, `bfloat16`, or `uint64`. The data layout can be ND. Non-contiguous tensors are supported.
- **`x_scale`** (`Tensor`): Required. Quantization factor for the activation matrix, $x\_scale$ in the formulas. This parameter must be 1D with shape `[m]`. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported.
- **`group_list`** (`Tensor`): Required. Number of tokens in each group involved in the computation, $groupList$ in the formulas. This parameter must be 1D with shape `[e]`, and its length must be identical to the first axis dimension of `weight`. The data type can be `int64`. The data layout can be ND. Non-contiguous tensors are supported.
- **`smooth_scale`** (`Tensor`): Optional. Smooth scaling factor, $smoothScale$ in the formulas. The data type can be `float32`. The data layout can be ND. This parameter must be provided only in A4W4 scenarios, and its first axis length must be identical to the first axis dimension of `weight`. The shape of this parameter is `(E, N/2)` or `(E,)`. When shape `(E,)` is used, broadcast multiplication is applied. In other scenarios, the default value is `None`.
- **`weight_assist_matrix`** (`TensorList`): Optional. Auxiliary matrix for the right matrix, $weightAssistMatrix$ in the formulas. The data type can be `float32`. The data layout can be ND. This parameter must be a 2D tensor. This parameter must be provided only in A8W4 scenarios, where the length of its first dimension must be identical to that of the first dimension of `weight`, and the length of its last dimension must be identical to that of the last dimension of `weight` when restored to the ND layout. In other scenarios, the default value is `None`.
- **`bias`** (`Tensor`): Optional. Offset value for matrix multiplication computation, $bias$ in the formulas. This parameter must be a 2D tensor. The data type can be `int32`. Currently, only the default value `None` is supported.
- **`dequant_mode`** (`int`): Optional. Dequantization mode. The data type can be `int32`. The default value is `0`. In A8W4 scenarios, valid values are `0` or `1`. In A8W8 and A4W4 scenarios, only `0` is supported.
    - `0`: enables the `per-token` activation matrix and `per-channel` weight matrix.
    - `1`: enables the `per-token` activation matrix and `per-group` weight matrix.
- **`dequant_dtype`** (`int`): Optional. Dequantization data type. The data type can be `int32`. Currently, only the default value `0` (indicating `float32`) is supported.
- **`quant_mode`** (`int`): Optional. Quantization mode after SwiGLU. The data type is `int32`. Currently, only the default value `0` (`per-token`) is supported.
- **`quant_dtype`** (`int`): Optional. Low-bit quantized data type after quantization. The data type can be `int32`. Currently, only the default value `0` (`int8`) is supported.
- **`group_list_type`** (`int`): Optional. Input type for `group_list`. The data type can be `int32`. The default value is `0`.
    - `0`: enables the `cumsum` mode, where each element in `group_list` indicates the cumulative length of the current group.
    - `1`: enables the `count` mode, where each element in `group_list` indicates the number of tokens contained in that group.
- **`tuning_config`** (`List[int]`): Optional. The first element specifies the expected number of tokens processed by each expert. Elements starting from the second element are reserved for future extensions and can be omitted. The default value is `None`.

## Return Values

- **`output`** (`Tensor`): Quantized output result, $Q$ in the formulas. The data type can be `int8`. This parameter must be 2D with shape `[m, n/2]`. The data layout can be ND. Non-contiguous tensors are supported.
- **`output_scale`** (`Tensor`): Output quantization factor, $Q\_scale$ in the formulas. The data type can be `float32`. This parameter must be 1D with shape `[m]`. The data layout can be ND. Non-contiguous tensors are supported.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- Deterministic computation: This API defaults to a deterministic implementation. For identical inputs, multiple execution passes generate identical outputs to guarantee repeatability.
- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
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
        |A8W4|(M, K)|The shape in ND layout `{(E, K, N)}` or the shape in NZ layout.|per-channel:{(E, N)}; per-group:{(E, K\_group\_num, N)}|(M,)|-|(M, N/2)|(M,)|
        |A4W4|(M, K)|The shape in ND layout `{(E, K, N)}` or the shape in NZ layout.|{(E, N)}|(M,)|`(E, N/2)` or `(E,)`|(M, N/2)|(M,)|

    - In A8W8 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 65536.
    - In A8W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.
    - In A4W4 scenarios, the size of the N axis must not exceed 10240, and the size of the last axis of `x` must be less than 20000.

## Examples

- Single-operator call

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

- Graph mode call

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
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weightscale, xscale, group_list, quant_dtype=quant_dtype)
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
    quant_dtype = 2
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu(), quant_dtype)
    ```
