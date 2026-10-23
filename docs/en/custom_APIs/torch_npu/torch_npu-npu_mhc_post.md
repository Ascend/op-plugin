# torch\_npu.npu\_mhc\_post

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- Function: performs Post Mapping on the output of the $l$-th layer $h_l^{out}$ and Res Mapping on the input of the $l$-th layer $x_l$ in the mHC (Manifold-Constrained Hyper-Connections) architecture, and then applies a residual connection to the two, obtaining the input of the $(l+1)$-th layer $x_{l+1}$. When `h_res` is `None`, Res Mapping is skipped and the residual addition is performed directly.
- Formulas:

    When `h_res` is provided:

    $$ x_{l+1} = (H_l^{res}) \times x_l + h_l^{out} \otimes H_l^{post} $$

    When `h_res` is omitted (passed as `None`):

    $$ x_{l+1} = x_l + h_l^{out} \otimes H_l^{post} $$

## Prototype

```python
torch_npu.npu_mhc_post(x, h_res, h_out, h_post) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Tensor to be processed, representing the input data of the mHC layer in the network. The data type can be `torch.bfloat16` or `torch.float16`. The shape can be `(B, S, n, D)` or `(T, n, D)`. The data layout can be `ND`. Non-contiguous tensors are supported. Empty tensors are supported.
- **`h_res`** (`Tensor`): Optional. The default value is `None`. The hRes transformation matrix of mHC, which is a doubly stochastic matrix after the Sinkhorn transformation. The data type can be `torch.float32`, the shape is (B, S, n, n) or (T, n, n), the data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported. When `None` is passed, Res Mapping is skipped and the formula degenerates to $x_{l+1} = x_l + h_l^{out} \otimes H_l^{post}$.
- **`h_out`** (`Tensor`): Required. Output of the Atten/MLP layer. The data type is the same as that of x, the shape is (B, S, D) or (T, D), the data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.
- **`h_post`** (`Tensor`): Required. The hPost transformation matrix of mHC. The data type can be `torch.float32`, the shape is (B, S, n) or (T, n), the data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.

## Return Values

**`y`** (`Tensor`): Required output. The output data of the mHC layer in the network, which serves as the input of the next layer. The data type is the same as that of `x`, the shape is (B, S, n, D) or (T, n, D), and the data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode calls.

## Example

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
    # When h_res is omitted:
    # y_npu = torch_npu.npu_mhc_post(x_npu, None, h_out_npu, h_post_npu)
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
