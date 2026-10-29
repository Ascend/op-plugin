# torch_npu.npu_clipped_swiglu

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

- Function: Swish-gated linear unit activation function with clipping, which implements the variant SwiGLU computation of `x`. Compared with torch_npu.npu_swiglu, this API adds some input parameters: `group_index`, `alpha`, `limit`, `bias`, and `interleaved`, to support the variant SwiGLU used by the GPT-OSS model and the grouping scenario used by MoE models.

- Formulas:

    For a given input tensor `x` with the dimensions [a,b,c,d,e,f,g...], the following computation is performed:

    1. Merge the axes of `x` based on the input parameter `dim`. The merged dimensions are [pre, cut, after]. The cut axis is the axis that must be split into two tensors after the axis merging, and the splitting method is either front-back splitting or odd-even splitting. pre and after can be 1. For example, when `dim` is 3, the dimensions of the merged `x` are [a \* b \* c, d, e \* f \* g \*...]. In addition, because the elements of the after axis are stored contiguously and the computation is element-wise, the cut axis and the after axis are merged, obtaining the dimensions of `x` as [pre, cut* after].

    2. Filter the pre axis of `x` according to the input parameter `group_index`. The formula is as follows:
        $$
        sum = \text{Sum}(group\_index)
        $$

        $$
        x = x[ : sum, : ]
        $$
        In the formula, sum indicates the sum of all elements of `group_index`. When `group_index` is not passed, this step is skipped.

    3. Split `x` according to the input parameter `interleaved`. The formula is as follows:

        When `interleaved` is True, odd-even splitting is used:
        $$
        A = x[ : , : : 2]
        $$

        $$
        B = x[ : , 1 : : 2]
        $$

        When `interleaved` is False, front-back splitting is used:
        $$
        h = x.shape[1] // 2
        $$

        $$
        A = x[ : , : h]
        $$

        $$
        B = x[ : , h : ]
        $$

    4. Perform the variant SwiGLU computation according to the input parameters `alpha`, `limit`, and `bias`. The formula is as follows:
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
    
    5. The number of dimensions of the reshaped output tensor y is the same as the number of dimensions of x before axis merging. The size on the `dim` axis is half that of `x`, and the other dimensions are the same as those of `x`.
    
## Prototype

```python
torch_npu.npu_clipped_swiglu(x, *, group_index=None, dim=-1, alpha=1.702, limit=7.0, bias=1.0, interleaved=True) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Target input tensor. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. Non-contiguous tensors are not supported. The data layout can be ND. `x` must have more than one dimension, and the size of the `dim` axis must be even.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`group_index`** (`Tensor`): Optional. Grouping configuration of `x`. This parameter must be 1D, where the $i$-th element represents the number of tokens in the $i$-th group of `x` after axis fusion. The data type can be `torch.int64`. The data layout can be ND. The default value is `None`, indicating that no grouping processing is applied to `x`.
- **`dim`** (`int`): Optional. Dimension index along which `x` is split. The value range is `[-x.dim(), x.dim() - 1]`. The default value is `-1`.
- **`alpha`** (`float`): Optional. GLU activation function coefficient. The default value is `1.702`.
- **`limit`** (`float`): Optional. Input threshold limit for the variant SwiGLU. The default value is `7.0`.
- **`bias`** (`float`): Optional. Bias in variant SwiGLU computation. The default value is `1.0`.
- **`interleaved`** (`bool`): Optional. Specifies whether to split the input `x` by using the odd-even method. Valid values are `True` (enables odd-even splitting) or `False` (enables front-back splitting). The default value is `True`.

## Return Values

`Tensor`

Represents `y` in the formulas, which is the output of the activation function. The data type is the same as that of the input `x`. In terms of dimensions, the `dim` dimension is 1/2 of that of the input `x`, and the other dimensions are the same as those of the input `x`. The data layout is $ND$.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and graph mode.

## Example

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
