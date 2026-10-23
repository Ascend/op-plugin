# torch_npu.npu_fast_gelu

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

- Function: Fast Gaussian Error Linear Units (FastGelu) activation function, which computes the forward result of `FastGelu` for each element of the input.
- Formulas:

  <!-- npu="910,310p" id6 -->
  - The formula for <term>Atlas inference products</term> and <term>Atlas training products</term> is as follows:

  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702 \mid x\mid}} e^{0.851 x(x- \mid x\mid)}
  $$
  <!-- end id6 -->
  <!-- npu="950,A3,910b" id7 -->
  - The formula for <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term> is as follows:

  $$
  fast\_gelu(x)=\frac{x}{1+e^{-1.702x}}
  $$
  <!-- end id7 -->
  
## Prototype

```python
torch_npu.npu_fast_gelu(self) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. $x$ in the formulas. The data layout can be $ND$. Non-contiguous tensors are supported. Up to 8 dimensions are supported. Empty tensors are supported.

  <!-- npu="910" id8 -->
  - <term>Atlas training products</term>: The data type can be `torch.float16` or `torch.float32`.
  <!-- end id8 -->
  <!-- npu="910b" id9 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`.
  <!-- end id9 -->
  <!-- npu="A3" id10 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`.
  <!-- end id10 -->
  <!-- npu="310p" id11 -->
  - <term>Atlas inference products</term>: The data type must be `torch.float16` or `torch.float32`.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.float32`, or `torch.bfloat16`.
  <!-- end id12 -->

## Return Values

`Tensor`

Represents the computation result of `fast_gelu`.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- The `self` input must not be `None`.

## Example

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
