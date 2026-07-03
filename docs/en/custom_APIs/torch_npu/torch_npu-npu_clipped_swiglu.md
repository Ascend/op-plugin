# torch_npu.npu_clipped_swiglu

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Implements a variant SwiGLU activation function with a truncated Swish gating linear unit. Compared with `torch_npu.npu_swiglu`, this API introduces additional input parameters: `group_index`, `alpha`, `limit`, `bias`, and `interleaved`, which are used to support the variant SwiGLU utilized by GPT-OSS models and the grouping scenarios utilized by MoE models.

- Formulas:

    For a given input tensor `x` with dimensions `[a, b, c, d, e, f, g, ...]`, the following operations are performed:

    1. The input `x` is collapsed along axes based on the input parameter `dim`. After axis fusion, the dimension becomes `[pre, cut, after]`. The `cut` axis represents the axis that needs to be split into two tensors, following either front-back splitting or odd-even splitting. The dimensions of `pre` and `after` can equal 1. For example, when `dim` is `3`, the dimension of `x` after axis fusion is `[a * b * c, d, e * f * g * ...]`. Additionally, because elements along the `after` axis are stored contiguously in memory and the operation is element-wise, the `cut` axis and `after` axis are fused, resulting in `x` having the shape `[pre, cut * after]`.

    2. Based on the input parameter `group_index`, the `pre` axis of `x` is filtered according to the formula:
        $$
        sum = \text{Sum}(group\_index)
        $$

        $$
        x = x[ : sum, : ]
        $$
        `sum` denotes the sum of all elements in `group_index`. If `group_index` is omitted, this step is skipped.

    3. Based on the input parameter `interleaved`, `x` is split as follows:

        When `interleaved` is `True`, it indicates odd-even splitting.
        $$
        A = x[ : , : : 2]
        $$

        $$
        B = x[ : , 1 : : 2]
        $$

        When `interleaved` is `False`, it indicates front-back splitting.
        $$
        h = x.shape[1] // 2
        $$

        $$
        A = x[ : , : h]
        $$

        $$
        B = x[ : , h : ]
        $$

    4. The variant SwiGLU computation is executed based on the input parameters `alpha`, `limit`, and `bias`.
        $$
        A = A.clamp(min=None, max=limit)
        $$
        
        $$
        B = B.clamp(min=-limit, max=limit)
        $$
        
        $$
        y\_glu = A * sigmoid(alpha * A)
        $$
        
        $$
        y = y\_glu * (B + bias)
        $$
    
    5. The output tensor `y` is reshaped to have the identical number of dimensions as `x` before axis fusion. The size of the `dim` axis is half that of `x`, and all other dimensions remain identical to `x`.
    
## Prototype

```python
torch_npu.npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Target input tensor. The data type can be `float16`, `bfloat16`, or `float32`. Non-contiguous tensors are not supported. The data layout can be ND. This parameter must have more than 1 dimension, and the size of the `dim` axis must be an even number.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`group_index`** (`Tensor`): Optional. Grouping configuration of `x`. This parameter must be 1D, where the $i$-th element represents the number of tokens in the $i$-th group of `x` after axis fusion. The data type can be `int64`. The data layout can be ND. The default value is `None`, indicating that no grouping processing is applied to `x`.
- **`dim`** (`int`): Optional. Dimension index along which `x` is split. The value range is `[-x.dim(), x.dim() - 1]`. The default value is `-1`.
- **`alpha`** (`float`): Optional. GLU activation function coefficient. The default value is `1.702`.
- **`limit`** (`float`): Optional. Input threshold limit for the variant SwiGLU. The default value is `7.0`.
- **`bias`** (`float`): Optional. Bias in variant SwiGLU computation. The default value is `1.0`.
- **`interleaved`** (`bool`): Optional. Specifies whether to split the input `x` by using the odd-even method. Valid values are `True` (enables odd-even splitting) or `False` (enables front-back splitting). The default value is `True`.

## Return Values

`Tensor`

Output of the activation function, $y$ in the formula. The data type must be identical to that of `x`. Regarding its shape, the dimension size along the `dim` axis is exactly half (`1/2`) that of `x`, while all other dimensions remain identical to those of `x`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports both single-operator mode and graph mode.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    tokens_num = 4608
    hidden_size = 2048
    x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
    group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
    y = torch_npu.npu_clipped_swiglu(
        x.npu(),
        group_index=group_index.npu(),
        dim=-1,
        alpha=1.702,
        limit=7.0,
        bias=1.0,
        interleaved=True
    )
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    device = torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    
    class ClippedSwigluModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, x, group_index, dim, alpha, limit, bias, interleaved):
            y = torch_npu.npu_clipped_swiglu(
                x.npu(),
                group_index=group_index.npu(),
                dim=dim,
                alpha=alpha,
                limit=limit,
                bias=bias,
                interleaved=interleaved
            )
            return y
    
    tokens_num = 4608
    hidden_size = 2048
    x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.float32)
    group_index = torch.randint(1, 101, (4, ), dtype=torch.int64)
    clipped_swiglu_model = ClippedSwigluModel().npu()
    clipped_swiglu_model = torch.compile(clipped_swiglu_model, backend=npu_backend, dynamic=True)
    y = clipped_swiglu_model(x, group_index, -1, 1.702, 7.0, 1.0, True)
    ```
