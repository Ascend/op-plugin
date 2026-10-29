# torch_npu.npu_swiglu_mx_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Applies the Swish gated linear unit (swiglu) activation function followed by DynamicMxQuant (dynamic MX quantization). It quantizes and outputs the input `x` after swiglu activation. The API supports MoE grouping scenarios and multiple swiglu computation modes.
- Formulas:

  Let the input be $x$. First, perform swiglu activation on the input to obtain $swigluOut$:

  - When `swiglu_mode=0` (traditional swiglu), taking `activate_left=True` as an example, $x$ is equally divided along the `activate_dim` axis into the first half $A$ and the second half $B$:

    $$
    swigluOut = Swish(A) \cdot B
    $$

    where:

    $$
    Swish(z) = z \cdot sigmoid(z)
    $$

  - When `swiglu_mode=1` (variant swiglu, GPT-OSS variant), $x$ is split into $x_{glu}$ (even indices) and $x_{linear}$ (odd indices) by parity, and `clamp_limit`, `glu_alpha`, and `glu_bias` are used for clipping and scaling:

    $$
    x_{glu} = clamp(x_{glu},\ max=clamp\_limit)
    $$

    $$
    x_{linear} = clamp(x_{linear},\ -clamp\_limit,\ clamp\_limit)
    $$

    $$
    out_{glu} = x_{glu} \cdot sigmoid(glu\_alpha \cdot x_{glu})
    $$

    $$
    swigluOut = out_{glu} \cdot (x_{linear} + glu\_bias)
    $$

  - When `swiglu_mode=2`, the computation is the same as that of `swiglu_mode=1` (both are variant swiglu and support `clamp_limit`, `glu_alpha`, and `glu_bias`). The difference is that $x_{glu}$ and $x_{linear}$ are split front-to-back (the same as the splitting method of `swiglu_mode=0`) instead of by parity.

  - When `swiglu_mode=3` (variant swiglu), $x$ is split front-to-back along the `activate_dim` axis into $x_{glu}$ and $x_{linear}$ (the same splitting method as `swiglu_mode=0`). Activation is performed before clipping, and `glu_alpha` and `glu_bias` are not used:

    $$
    x_{glu} = x_{glu} \cdot sigmoid(x_{glu})
    $$

    $$
    x_{glu} = clamp(x_{glu},\ max=clamp\_limit)
    $$

    $$
    x_{linear} = clamp(x_{linear},\ -clamp\_limit,\ clamp\_limit)
    $$

    $$
    swigluOut = x_{glu} \cdot x_{linear}
    $$

  Then DynamicMxQuant quantization is performed on $swigluOut$:

  - When `scale_alg=0`, $swigluOut$ is grouped on the `axis` dimension by $k=blocksize=32$ values, and each group $\{V_i\}_{i=1}^{k}$ is dynamically quantized to $\{mxscale,\ \{P_i\}_{i=1}^{k}\}$:

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i / mxscale,\ round\_mode)
    $$

    Here, $emax$ is the exponent bit of the maximum normal number of the target data type.

    |   Data type   | emax |
    |:------------:|:----:|
    | `torch_npu.float4_e2m1fn_x2`  |  2   |
    | `torch_npu.float4_e1m2fn_x2`  |  0   |
    | `torch.float8_e4m3fn`|  8   |
    | `torch.float8_e5m2`  |  15  |

  - When `scale_alg=1` (only the two float8 types are involved), the scaling factor is calculated by block, and the maximum absolute value in the block is found:

    $$
    Amax(D^b) = max(\{|d_i|\}_{i=1}^{k})
    $$

    $$
    S_{fp32}^b = \frac{Amax(D^b)}{Amax(DType)}
    $$

    Convert $S_{fp32}^b$ to $S_{ue8m0}^b = 2^{E_{int}^b}$, which is representable in FP8, and quantize the elements in the block:

    $$
    d^i = DType(d_{fp32}^i \cdot R_{fp32}^b),\quad R_{fp32}^b = \frac{1}{fp32(S_{ue8m0}^b)}
    $$

## Prototype

```python
torch_npu.npu_swiglu_mx_quant(x, *, group_index=None, activate_dim=-1, activate_left=False, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0, group_mode=0, axis=-1, dst_type=296, round_mode="rint", scale_alg=0, max_dtype_value=0) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input data to be processed. The shape is $[X_1,X_2,\dots,X_n,2H]$, with 2 to 7 dimensions. The dimension corresponding to the `activate_dim` axis must be a multiple of 2. The data type can be `torch.float16` or `torch.bfloat16`. The data layout is ND.
- **`*`**: Syntax separator used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`group_index`** (`Tensor`): Optional. `group_index` required for MoE grouping. It must be a 1D Tensor with shape $[groupNum]$ ($groupNum$ ranges from 1 to 256). The data type can be `torch.int64`. The data layout is ND. The default value is `None`, indicating that no grouping is performed.
- **`activate_dim`** (`int`): Optional. Splitting axis selected for the Swish computation. The value range is $[-1,-2,x.dim()-2,x.dim()-1]$. The default value is `-1`.
- **`activate_left`** (`bool`): Optional. Whether to apply Swish activation to the left half after splitting. When the value is `True`, activation is applied to the left half. When the value is `False`, activation is applied to the right half. When `swiglu_mode=1`, activation is applied to the even block by default. The default value is `False`.
- **`swiglu_mode`** (`int`): Optional. swiglu computation mode. The value range is [0, 3]: `0` indicates traditional swiglu; `1` indicates variant swiglu (GPT-OSS variant), which uses parity splitting and supports `clamp_limit`, `glu_alpha`, and `glu_bias`; `2` uses the same computation as `1` but uses front-to-back splitting (same as `0`); `3` indicates variant swiglu, with the same splitting method as `2`, except that activation is performed first (the sigmoid coefficient is fixed to 1) and then clipping is performed, and `glu_alpha` and `glu_bias` are not supported. The default value is `0`. When `activate_dim` is not the last axis, `swiglu_mode` must be `0`.
- **`clamp_limit`** (`float`): Optional. Input threshold for variant swiglu, used to clip the input. It must be greater than 0 and finite. The default value is `7.0`.
- **`glu_alpha`** (`float`): Optional. Coefficient of the GLU activation function. The default value is `1.702`.
- **`glu_bias`** (`float`): Optional. Bias in the swiglu computation. The default value is `1.0`.
- **`group_mode`** (`int`): Optional. Mode corresponding to `group_index`. The value range is [0, 1]: `0` indicates count mode, and `1` indicates cumsum mode. The default value is `0`.
- **`axis`** (`int`): Optional. Axis on which DynamicMxQuant quantization occurs. The value range is $[-1,-2,x.dim()-2,x.dim()-1]$. The default value is `-1`.
- **`dst_type`** (`int`): Optional. Data type of the output `y`. `296` indicates `torch_npu.float4_e2m1fn_x2`, `297` indicates `torch_npu.float4_e1m2fn_x2`, `292` indicates `torch.float8_e4m3fn`, and `291` indicates `torch.float8_e5m2`. The default value is `296`.
- **`round_mode`** (`str`): Optional. Rounding mode of the output `y`. The value can be `"rint"`, `"round"`, or `"floor"`. When `dst_type` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, only `"rint"` is supported. The default value is `"rint"`.
- **`scale_alg`** (`int`): Optional. Calculation method of `mxscale`. The value range is [0, 2]: `0` represents the OCP algorithm, corresponding to the `scale_alg=0` scenario; `1` represents the cuBLAS algorithm, corresponding to the `scale_alg=1` scenario; `2` represents the RNE algorithm and is a reserved value. Currently, only `0` and `1` are supported. When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, only `0` is supported. The default value is `0`.
- **`max_dtype_value`** (`float`): Optional. Reserved parameter. Maximum value of the target data type specified during DynamicMxQuant. The value must be not less than 0. It takes effect only when `scale_alg=2` and `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`. The default value is `0`.

## Return Values

- **`y`** (`Tensor`): Quantized output, corresponding to $P_i$ or $d^i$ in the formulas. The data type is determined by `dst_type`. The data layout is ND. When `activate_dim` is the last axis, the shape is $[X_1,\dots,X_n,H]$ (that is, half of the input on the `activate_dim` axis). When `activate_dim` is not the last axis, the `activate_dim` axis is half of the corresponding input axis.
- **`mxscale`** (`Tensor`): Quantization scale corresponding to each group, corresponding to $mxscale$ or $S^b$ in the formulas. The data type is `torch.float8_e8m0fnu`. The data layout is ND.

## Constraints

- This API supports inference scenarios only.
- This API supports Eager mode and graph mode.
- The dimension of the input `x` corresponding to the `activate_dim` axis must be a multiple of 2, and the number of dimensions of `x` must be greater than 1.
- When `activate_dim` is not the last axis, `swiglu_mode` must be `0`.
- When `swiglu_mode` is `2` or `3`, `axis` must be `-1`.
- When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last dimension of `y` must be a multiple of 2, and `scale_alg` must be `0`.
- When `dst_type` is `torch.float8_e4m3fn` or `torch.float8_e5m2`, `round_mode` must be `"rint"`.
- The sum of all elements in `group_index` must not be greater than the product of the remaining axes of the input `x` except the last axis, and each element must be greater than 0.
- When `activate_dim` or `axis` is not the last axis and `group_index` exists, `x` must be 2D.
- The part of the output `y` and `mxscale` that exceeds the sum of all elements in `group_index` is not cleared, and the memory in this part contains garbage data.

## Examples

```python
import torch
import torch_npu

x = torch.randn(2, 64, dtype=torch.bfloat16).npu()
group_index = torch.tensor([2], dtype=torch.int64).npu()

y, mxscale = torch_npu.npu_swiglu_mx_quant(
    x,
    group_index=group_index,
    activate_dim=-1,
    activate_left=True,
    swiglu_mode=1,
    clamp_limit=7.0,
    glu_alpha=1.0,
    glu_bias=1.702,
    group_mode=0,
    axis=-1,
    dst_type=torch_npu.float4_e2m1fn_x2,
    round_mode="rint",
    scale_alg=0,
    max_dtype_value=0,
)

print("y:", y)
print("mxscale:", mxscale)
```
