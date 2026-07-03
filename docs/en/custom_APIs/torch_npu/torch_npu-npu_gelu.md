# torch_npu.npu_gelu

> [!NOTICE]  
> In the NPU environment, the native Torch API `gelu` ignores the `approximate` parameter and defaults to `tanh`. Use this API to set `approximate` to `none` or to select different approximation modes.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas training products</term>  | √   |

## Function

- Description: Computes the Gaussian Error Linear Unit (GELU) activation function.
- Formulas:

    The GELU expression is as follows:
    $$
    \text{GELU}(x) = x \cdot \Phi(x)
    $$

    $\Phi(x)$ indicates the cumulative distribution function (CDF) of the Gaussian distribution. The expression is as follows:
    $$
    \Phi(x) = P(X \leq x) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]
    $$
    When `approximate="none"`, this expression is used. When `approximate="tanh"`, $\text{erf}(\cdot)$ is replaced with a $\tanh(\cdot)$-based approximation.

## Prototype

```python
torch_npu.npu_gelu(input, approximate='none') -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor, $x$ in the formulas. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 8D.
    - <term>Atlas training products</term>: The data type can be `float16` or `float32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float32`, `float16`, or `bfloat16`.
    - <term>Atlas inference products</term>: The data type can be `float16` or `float32`.

- **`approximate`** (`str`): Optional. Activation function mode used for calculation, which can be set to `"none"` or `"tanh"`. `"none"` indicates `erf` mode and `"tanh"` indicates `tanh` mode.

## Return Values

`Tensor`

The data type must be identical to `input`. The data layout can be ND. The shape must be identical to that of `input`. Non-contiguous tensors are supported.

## Constraints

- This API supports graph mode.
- The `input` parameter must not be `None`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> input_tensor = torch.randn(100, 200)
    >>> output_tensor = torch_npu.npu_gelu(input_tensor.npu(), approximate='tanh')
    >>>
    >>> print(output_tensor)
    tensor([[ 0.5795, -0.0274, -0.1477,  ...,  0.2422, -0.0843, -0.1154],
            [-0.0385,  0.8736,  0.1809,  ..., -0.0676,  0.2404,  0.4038],
            [ 0.0438,  0.0205, -0.1536,  ...,  0.2910,  1.1553,  0.3319],
            ...,
            [-0.1698, -0.0031,  0.5120,  ..., -0.1390, -0.0082,  0.6286],
            [ 0.1980,  0.0535, -0.1685,  ..., -0.1528, -0.1484,  1.0703],
            [-0.1351,  1.5851, -0.0222,  ..., -0.0230,  1.4319, -0.1700]],
        device='npu:0')
    >>> print(output_tensor.shape)
    torch.Size([100, 200])
    >>> print(output_tensor.dtype)
    torch.float32
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Net(torch.nn.Module):
    
        def __init__(self):
            super().__init__()
        def forward(self, self_0, approximate):
            out = torch_npu.npu_gelu(self_0, approximate=approximate)
            return out
    
    x = torch.randn(100, 10, 20).npu()
    model = Net()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = model(x, approximate="none")    
    print(npu_out.shape, npu_out.dtype)

    # Expected output of the preceding code sample:
    torch.Size([100, 10, 20]) torch.float32
    ```
