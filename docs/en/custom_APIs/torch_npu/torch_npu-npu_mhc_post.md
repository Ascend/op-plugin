# torch\_npu.npu\_mhc\_post

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- Performs post mapping on the output h<sub>l</sub><sup>out</sup> of layer l in the Manifold-Constrained Hyper-Connections (mHC) architecture and Res mapping on the input x<sub>l</sub> of layer l, and then applies a residual connection to the two results to obtain the input x<sub>l+1</sub> of layer l+1.

- Formula:

    $$ x_{l+1} = (H_l^{res}) \times x_l + h_l^{out} \otimes H_l^{post} $$
    
## Prototype

```python
torch_npu.npu_mhc_post(x, h_res, h_out, h_post) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Tensor to be processed, representing the input data of the mHC layer in the network. The data type can be `bfloat16` or `float16`. The shape can be `(B, S, n, D)` or `(T, n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`h_res`** (`Tensor`): Required. mHC `h_res` transformation matrix, which is the doubly stochastic matrix obtained after the Sinkhorn transformation. The data type can be `float32`. The shape can be `(B, S, n, n)` or `(T, n, n)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`h_out`** (`Tensor`): Required. Output of the Attention/MLP layer. The data type is the same as that of `x`. The shape can be `(B, S, D)` or `(T, D)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`h_post`** (`Tensor`): Required. mHC `h_post` transformation matrix. The data type can be `float32`. The shape can be `(B, S, n)` or `(T, n)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are supported.

## Return Values

**`y`** (`Tensor`): Required output. Output data of the mHC layer in the network, serving as the input to the next layer. The data type is the same as that of `x`. The shape can be `(B, S, n, D)` or `(T, n, D)`. The data layout can be `ND`.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator and TorchAir graph mode calls.

## Examples

- Single-operator call

    ```python
    import torch 
    import torch_npu
    import numpy as np
    print(torch.npu.is_available())
    # Check NPU availability
    assert torch.npu.is_available(), "NPU not available"
    print("get npu number.")
    num_npus = torch.npu.device_count()
    print("Number of NPUs:", num_npus)
    x_shape = [1,4,512]
    h_res_shape = [1,4,4]
    h_out_shape = [1,512]
    h_post_shape = [1,4]
    x = torch.rand(x_shape, dtype=torch.float16)
    h_res = torch.rand(h_res_shape, dtype=torch.float32)
    h_out = torch.rand(h_out_shape, dtype=torch.float16)
    h_post = torch.rand(h_post_shape, dtype=torch.float32)
    x_npu = x.npu()
    h_res_npu = h_res.npu()
    h_out_npu = h_out.npu()
    h_post_npu = h_post.npu()
    y_npu = torch_npu.npu_mhc_post(x_npu, h_res_npu, h_out_npu, h_post_npu)
    ```

- TorchAir graph mode call (aclgraph)

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    from torchair.core.utils import logger
    import os
    import logging
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    device=torch.device(f'npu:0')
    torch_npu.npu.set_device(device)
    class MhcPostModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, h_res, h_out, h_post):
            y = torch.ops.npu.npu_mhc_post(x, h_res, h_out, h_post)
            return y
    x_shape = [1,1,4,512]
    h_res_shape = [1,1,4,4]
    h_out_shape = [1,1,512]
    h_post_shape = [1,1,4]
    x = torch.rand(x_shape, device='npu', dtype=torch.float16)
    h_res = torch.rand(h_res_shape, device='npu', dtype=torch.float32)
    h_out = torch.rand(h_out_shape, device='npu', dtype=torch.float16)
    h_post = torch.rand(h_post_shape, device='npu', dtype=torch.float32)
    mhc_post_model = MhcPostModel().npu()
    mhc_post_model = torch.compile(mhc_post_model, backend=npu_backend)
    y = mhc_post_model(x, h_res, h_out, h_post)
    ```
