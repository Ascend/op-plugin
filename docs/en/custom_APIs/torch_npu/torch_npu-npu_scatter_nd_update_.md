# torch_npu.npu_scatter_nd_update_

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

Updates the values in `input` with the values in `updates` according to the specified `indices`, and saves the result to the output tensor. The data in `input` is changed.

## Prototype

```python
torch_npu.npu_scatter_nd_update_(input, indices, updates) -> Tensor(a!)
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `updates`. It must have 1 to 8 dimensions.

  <!-- npu="310p" id6 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bool`.
  <!-- end id6 -->
  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bool`.
  <!-- end id7 -->
  <!-- npu="910b" id8 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, or `torch.int8`.
  <!-- end id8 -->
  <!-- npu="A3" id9 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, or `torch.int8`.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, `torch.int8`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`.
  <!-- end id10 -->

- **`indices`** (`Tensor`): Required. Index tensor. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. Non-contiguous tensors are supported. The index values in `indices` must not be out of bounds.
- **`updates`** (`Tensor`): Required. Update data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `input`.

  <!-- npu="310p" id11 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bool`.
  <!-- end id11 -->
  <!-- npu="910" id12 -->
  - <term>Atlas training products</term>: The data type can be `torch.float32`, `torch.float16`, or `torch.bool`.
  <!-- end id12 -->
  <!-- npu="910b" id13 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, or `torch.int8`.
  <!-- end id13 -->
  <!-- npu="A3" id14 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, or `torch.int8`.
  <!-- end id14 -->
  <!-- npu="950" id15 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`, `torch.float16`, `torch.bool`, `torch.bfloat16`, `torch.int64`, `torch.int8`, `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch_npu.hifloat8`.
  <!-- end id15 -->

## Return Values

`Tensor`

The result of updating `input`.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- `indices` must be at least 2D, and the size of its last dimension must not exceed the number of dimensions of `input`.
- If the size of the last dimension of `indices` is a, the shape of `updates` equals the shape of `indices` excluding the last dimension plus the shape of `input` excluding the first a dimensions. For example, if the shape of `input` is $(4, 5, 6)$ and the shape of `indices` is $(3, 2)$, the shape of `updates` must be $(3, 6)$.

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    
    data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
    var = torch.from_numpy(data_var).to(torch.float16).npu()
       
    data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
    indices = torch.from_numpy(data_indices).to(torch.int32).npu()
    
    data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
    updates = torch.from_numpy(data_updates).to(torch.float16).npu()
    
    out = torch_npu.npu_scatter_nd_update_(var, indices, updates)
    print(out)
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[1.8271, 1.4551, 1.3154,  ..., 1.9854, 1.4365, 1.0732],
            [1.9492, 1.6455, 1.6504,  ..., 1.5957, 1.6201, 1.4385],
            [0.0742, 0.1982, 0.8945,  ..., 0.4912, 0.6753, 0.1120],
            ...,
            [0.1113, 0.6255, 0.7686,  ..., 0.0247, 0.2490, 0.6909],
            [0.4312, 0.7954, 0.7339,  ..., 0.1154, 0.6440, 0.3342],
            [0.9570, 0.2869, 0.6489,  ..., 0.7451, 0.0234, 0.8843]],
            device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    import os
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch.nn as nn
    import torch
    import numpy as np
    import numpy
    torch_npu.npu.set_compile_mode(jit_compile=True)
    
    os.environ["ENABLE_ACLNN"] = "false"
    
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()
    
        def forward(self, var, indices, update):
            # Call the target API
            res = torch_npu.npu_scatter_nd_update_(var, indices, update)
            return res
            
    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
    
    dtype = np.float32
    x = [33 ,5]
    indices = [33,25,1]
    update = [33,25,5]
    
    data_x = np.random.uniform(0, 1, x).astype(dtype)
    data_indices = np.random.uniform(0, 10, indices).astype(dtype)
    data_update = np.random.uniform(0, 1, update).astype(dtype)
    
    tensor_x = torch.from_numpy(data_x).to(torch.float16)
    tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
    tensor_update = torch.from_numpy(data_update).to(torch.float16)
    
    # Pass the parameters
    out= npu_mode (tensor_x.npu(), tensor_indices.npu(), tensor_update.npu())
    print(out.shape, out.dtype)

    # Expected output of the preceding code sample:
    torch.Size([33, 5]) torch.float16
    ```
