# torch_npu.npu_ffn

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

- Description: Provides Mixture-of-Experts Feed-Forward Network (MoeFFN) and Feed-Forward Network (FFN) computation features. FFN is used when there are no expert groups (`expert_tokens` is empty) and MoeFFN is used when there are expert groups (`expert_tokens` is not empty).
- Formula:

     $$\text{activation}$$ is the activation function used. $$W_1$$ and $$W_2$$ correspond to `weight1` and `weight2` of the input parameters, respectively. $$b_1$$ and $$b_2$$ correspond to `bias1` and `bias2` of the input parameters, respectively.

    - Non-quantization scenarios:
        $$
        y=activation(x * W1 + b1) * W2 + b2
        $$

    - Quantization scenarios:
        $$
        y=((activation((x * W1 + b1) * deq\_scale1) * scale + offset) * W2 + b2) * deq\_scale2
        $$

    - Fake-quantization scenarios:
        $$
        y=activation(x * ((W1 + antiquant\_offset1) * antiquant\_scale1) + b1) * ((W2 + antiquant\_offset2) * antiquant\_scale2) + b2
        $$

> [!NOTE]  
> When the activation function is `geglu`, `swiglu`, or `reglu`, enabling FFN must meet the following threshold requirements. Specifically, the FFN fusion operator is recommended only when the vector execution time of the corresponding small operators across the entire network takes at least 30 μs and accounts for more than 10% of the total runtime. Alternatively, if the performance of the small operators is unknown, try enabling the fusion operator. If performance deteriorates, disable it.

## Prototype

```python
torch_npu.npu_ffn(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=None, output_dtype=None) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor, $x$ in the formulas. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. The input tensor must have at least two dimensions $[M, K1]$ and at most eight dimensions.

- **`weight1`** (`Tensor`): Required. Expert weight data, $W1$ in the formulas. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. The input shapes with and without experts are `[E, K1, N1]` and `[K1, N1]`, respectively.

- **`weight2`** (`Tensor`): Required. Expert weight data, $W2$ in the formulas. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. The input shapes with and without experts are `[E, K2, N2]` and `[K2, N2]`, respectively.

    > [!NOTE]  
    > $M$ indicates the number of tokens, corresponding to $B$ (`Batch`) and $S$ (`Seq-Length`) in the Transformer. $K1$ indicates the number of input channels of the first MatMul, corresponding to $H$ (`Head-Size`, size of the hidden layer) in the Transformer. $N1$ indicates the number of output channels of the first MatMul. $K2$ indicates the number of input channels of the second MatMul. $N2$ indicates the number of output channels of the second MatMul, corresponding to $H$ in the Transformer. $E$ indicates the number of experts in expert scenarios.

- **`activation`** (`str`): Required. The activation function used. Currently, only `fastgelu`, `gelu`, `relu`, `silu`, `geglu`, `swiglu`, and `reglu` are supported.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`expert_tokens`** (`list`): Optional. Number of tokens for each expert. The data type can be `int32`. The data layout can be ND. If this parameter is not empty, its maximum supported length is 256.
- **`expert_tokens_index`** (`list`): Optional. Token indices calculated by each expert. The data type can be `int32`. The data layout can be ND. If this parameter is not empty, its maximum supported length is 256.

- `bias1` (`Tensor`): Optional. Weight data correction value, $b1$ in the formulas. The data type can be `float16`, `float32`, or `int32`. The data layout can be ND. The input shapes with and without experts are `[E, N1]` and `[N1]`, respectively.
- **`bias2`** (`Tensor`): Optional. Weight data correction value, $b2$ in the formulas. The data type can be `float16`, `float32`, or `int32`. The data layout can be ND. The input shapes with and without experts are `[E, N2]` and `[N2]`, respectively.

- **`scale`** (`Tensor`): Optional. Quantization parameter, quantization scaling factor. The data type can be `float32`. The data layout can be ND. In `pertensor` mode, the input is a 1D vector with shapes of `[E]` and `[1]` with and without experts, respectively. In `perchannel` mode, the input is a 2D vector or a 1D vector with shapes of `[E, N1]` and `[N1]` with and without experts, respectively.
- **`offset`** (`Tensor`): Optional. Quantization parameter, quantization offset. The data type can be `float32`. The data layout can be ND. It is a 1D vector with shape `[E]` or `[1]` with and without experts, respectively.
- **`deq_scale1`** (`Tensor`): Optional. Quantization parameter, dequantization scaling factor of the first MatMul group. The data type can be `int64`, `float32`, or `bfloat16`. The data layout can be ND. The input shapes with and without experts are `[E, N1]` and `[N1]`, respectively.
- **`deq_scale2`** (`Tensor`): Optional. Quantization parameter, dequantization scaling factor of the second MatMul group. The data type can be `int64`, `float32`, or `bfloat16`. The data layout can be ND. The input shapes with and without experts are `[E, N2]` and `[N2]`, respectively.
- **`antiquant_scale1`** (`Tensor`): Optional. Fake-quantization parameter, scaling factor of the first MatMul group. The data type can be `float16` or `bfloat16`. The data layout can be ND. In `perchannel` mode, the input shapes with and without experts are `[E, N1]` and `[N1]`, respectively.
- **`antiquant_scale2`** (`Tensor`): Optional. Fake-quantization parameter, scaling factor of the second MatMul group. The data type can be `float16` or `bfloat16`. The data layout can be ND. In `perchannel` mode, the input shapes with and without experts are `[E, N2]` and `[N2]`, respectively.
- **`antiquant_offset1`** (`Tensor`): Optional. Fake-quantization parameter, offset of the first MatMul group. The data type can be `float16` or `bfloat16`. The data layout can be ND. In `perchannel` mode, the input shapes with and without experts are `[E, N1]` and `[N1]`, respectively.
- **`antiquant_offset2`** (`Tensor`): Optional. Fake-quantization parameter, offset of the second MatMul group. The data type can be `float16` or `bfloat16`. The data layout can be ND. In `perchannel` mode, the input shapes with and without experts are `[E, N2]` and `[N2]`, respectively.

- **`inner_precise`** (`int`): Optional. Choice between high accuracy and high performance. The data type can be `int64`. This parameter takes effect only for `float16`. `bfloat16` and `int8` do not distinguish between high-precision and high-performance modes.

    - When `inner_precise` is set to `0`, high-precision mode is enabled, and the operator uses the `float32` data type internally for computation.
    - When `inner_precise` is set to `1`, high performance mode is enabled.

  In `bfloat16` non-quantization scenarios, `inner_precise` can only be set to `0`. In `float16` non-quantization scenarios, it can be set to `0` or `1`. In quantization or fake-quantization scenarios, `inner_precise` can be set `0` or `1`, but the setting does not take effect.

- **`output_dtype`** (`ScalarType`): Optional. Data type of the output tensor. This parameter takes effect only in quantization scenarios. The data type can be `float16` or `bfloat16`. The default value is `None`, indicating that the data type of the output tensor is `float16`.

## Return Values

`Tensor`

Output tensor, $y$ in the formula. The data type can be `float16` or `bfloat16`. The data layout can be ND. The number of output dimensions must be identical to that of `x`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- If there are experts, the total number of experts must match $M$ of `x`.
- When the activation layer is `geglu`, `swiglu`, or `reglu`, only the `float16` high-performance scenario (the data types of all mandatory `Tensor` parameters are `float16`) is supported without expert grouping. Here, $N1=2*K2$.
- When the activation function is `gelu`, `fastgelu`, `relu`, or `silu`, the following scenarios are supported, with or without expert grouping: `float16` high-precision, `float16` high-performance, `bfloat16`, quantization, and fake-quantization. Here, $N1=K2$.
- In all scenarios, the following general dimension constraints must be met: $K1=N2$, $K1<65536$, $K2<65536$, and the $M$ dimension must be less than the maximum value of `int32` after 32-byte alignment. Additionally, the relationship between $N1$ and $K2$ is determined by the activation function type. `geglu`, `swiglu`, and `reglu` require $N1=2*K2$, while `gelu`, `fastgelu`, `relu`, and `silu` require $N1=K2$. 
- Quantization parameters and fake-quantization parameters cannot be passed in non-quantization scenarios. Fake-quantization parameters cannot be passed in quantization scenarios. Quantization parameters cannot be passed in fake-quantization scenarios.
- Parameter data types in quantization scenarios: `x` is `int8`, `weight` is `int8`, `bias` is `int32`, `scale` is `float32`, and `offset` is `float32`. Other parameters depend on the type of `y`:
    - When `y` is `float16`, the data type `deq_scale` can be `uint64`, `int64`, or `float32`.
    - When `y` is `bfloat16`, `deq_scale` can be `bfloat16`.
    - The data types of `deq_scale1` and `deq_scale2` must be the same.

- Parameter data types in quantization scenarios supporting the `perchannel` mode for `scale`: `x` is `int8`, `weight` is `int8`, `bias` is `int32`, `scale` is `float32`, and `offset` is `float32`. Other parameters depend on the type of `y`:
    - When `y` is `float16`, `deq_scale` can be `uint64` or `int64`.
    - When `y` is `bfloat16`, `deq_scale` can be `bfloat16`.
    - The data types of `deq_scale1` and `deq_scale2` must be the same.

- Fake-quantization scenarios support two parameter configurations:
    - `y` is `float16`, `x` is `float16`, `bias` is `float16`, `antiquant_scale` is `float16`, `antiquant_offset` is `float16`, and `weight` can be `int8`.
    - `y` is `bfloat16`, `x` is `bfloat16`, `bias` is `float32`, `antiquant_scale` is `bfloat16`, `antiquant_offset` is `bfloat16`, and `weight` can be `int8`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn((1, 1280), device='npu', dtype=torch.float16)
    >>> weight1 = torch.randn(1280, 10240, device='npu', dtype=torch.float16)
    >>> weight2 = torch.randn(10240, 1280, device='npu', dtype=torch.float16)
    >>> activation = "fastgelu"
    >>> npu_out = torch_npu.npu_ffn(x, weight1, weight2, activation, inner_precise=1)
    >>>
    >>> print(npu_out)
    tensor([[ 1474.0000,  2000.0000,  1683.0000,  ...,  1938.0000, -1353.0000,
            207.8750]], device='npu:0', dtype=torch.float16)
    >>>
    >>> print(npu_out.shape)
    torch.Size([1, 1280])
    ```

- Graph mode call

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

    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight1, weight2, activation, expert):
            return torch_npu.npu_ffn(x, weight1, weight2, activation,  expert_tokens=expert, inner_precise=1)

    model = MyModel().npu()
    x = torch.randn((1954, 2560), device='npu', dtype=torch.float16)
    weight1 = torch.randn((16, 2560, 5120), device='npu', dtype=torch.float16)
    weight2 = torch.randn((16, 5120, 2560), device='npu', dtype=torch.float16)
    activation = "fastgelu"
    expert = [227, 62, 78, 126, 178, 27, 122, 1, 19, 182, 166, 118, 66, 217, 122, 243]
    model = torch.compile(model, backend=npu_backend, dynamic=True)

    npu_out = model(x, weight1, weight2, activation, expert)
    print(npu_out.shape)
    print(npu_out)

    # Expected output of the preceding code sample:
    torch.Size([1954, 2560])
    tensor([[  736.5000,  2558.0000,  3806.0000,  ..., -4180.0000,  -707.5000,
            1692.0000],
            [  113.0000,  1471.0000,  2492.0000,  ...,   404.5000, -1629.0000,
            -881.0000],
            [-3046.0000,  -401.0000,  3780.0000,  ...,  -518.5000,  -151.1250,
            3962.0000],
            ...,
            [ 2694.0000, -4648.0000,   -23.4844,  ..., -2624.0000, -2112.0000,
            -1070.0000],
            [ -438.0000, -3500.0000,  -941.0000,  ..., -2626.0000, -3878.0000,
            -2076.0000],
            [-2194.0000, -1583.0000, -1336.0000,  ...,  3906.0000,  -222.7500,
            -58.9688]], device='npu:0', dtype=torch.float16)

    ```
