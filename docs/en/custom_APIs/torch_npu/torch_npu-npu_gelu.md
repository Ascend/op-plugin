# torch_npu.npu_gelu

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.<br>
> When the native PyTorch `gelu` API is used in the NPU environment, the `approximate` parameter does not take effect and defaults to `tanh`. Use this API if you need to set the default value of `approximate` to `None` or to distinguish between different `approximate` methods.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

- Function: computes the Gaussian Error Linear Unit (GELU) activation function.
- Formulas:

    The expression of Gaussian Error Linear Unit (GELU) is:
    $$
    \text{GELU}(x) = x \cdot \Phi(x)
    $$

    $\Phi(x)$ is the CDF (cumulative distribution function) of the Gaussian distribution, and its expression is:
    $$
    \Phi(x) = P(X \leq x) = \frac{1}{2} \left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]
    $$
    When `approximate="none"`, the preceding formula is used. When `approximate="tanh"`, $\text{erf}(\cdot)$ is replaced by the approximate form of $\tanh(\cdot)$.

## Prototype

```python
torch_npu.npu_gelu(self, *, approximate='none') -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Input to be computed by `npu_gelu`, $x$ in the formula. The data layout can be $ND$. Non-contiguous tensors are supported. Empty tensors are supported. Up to 8 dimensions are supported.

    <!-- npu="910,310p" id5 -->
    - <term>Atlas training products</term> and <term>Atlas inference products</term>: The data type can be `torch.float16` or `torch.float32`.
    <!-- end id5 -->
    <!-- npu="950,910b" id6 -->
    - <term>Atlas A2 products</term> and <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
    <!-- end id6 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`approximate`** (`str`): Optional. Activation function mode used for calculation, which can be set to `"none"` or `"tanh"`. `"none"` indicates `erf` mode and `"tanh"` indicates `tanh` mode.

## Return Values

`Tensor`

The data type and shape must be the same as those of `self`. The data layout can be $ND$. Non-contiguous tensors are supported.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- The `self` input must not be `None`.

## Example

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
    # Graph mode approach
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
