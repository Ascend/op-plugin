# torch_npu.npu_fast_gelu

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

- Description: Computes the forward result of the Fast Gaussian Error Linear Units (`FastGelu`) activation function for each element in the input tensor.

- Formulas:

  - Formula for Atlas training products and Atlas inference products:
  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702 \mid x\mid}} e^{0.851 x(x- \mid x\mid)}
  $$

  - Formula for Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products:
  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702x}}
  $$

## Prototype

```python
torch_npu.npu_fast_gelu(input) -> Tensor
```

## Parameters

**`input`** (`Tensor`): $x$ in the formula. The data layout can be ND. Non-contiguous tensors are supported. This parameter can be up to 8D. Empty tensors are supported.

- Atlas training products: The data type can be `float16` or `float32`.
- Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `float32`, or `bfloat16`.
- Atlas A3 training products/Atlas A3 inference products: The data type can be `float16`, `float32`, or `bfloat16`.
- Atlas inference products: The data type must be `float16` or `float32`.

## Return Values

`Tensor`

Indicates the computation result of `fast_gelu`.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- The `input` parameter must not be `None`.

## Examples

- Single-operator call

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np

    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y = torch_npu.npu_fast_gelu(x).cpu().numpy()
    ```

- Graph mode call

    ```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    
    os.environ["ENABLE_ACLNN"] = "false"
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
        def forward(self, x): 
            y = torch_npu.npu_fast_gelu(x)
            return y
            
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    data_var = np.random.uniform(0, 1, [4, 2048, 16, 128]).astype(np.float32)
    x = torch.from_numpy(data_var).to(torch.float32).npu()
    y =npu_mode(x).cpu().numpy()
    print("shape of y:",y.shape)
    print("dtype of y:",y.dtype)
    
    # Expected output of the preceding code sample:
    shape of y: (4, 2048, 16, 128)
    dtype of y: float32
    ```
