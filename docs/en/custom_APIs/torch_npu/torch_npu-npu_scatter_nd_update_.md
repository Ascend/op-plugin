# torch_npu.npu_scatter_nd_update_

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Updates the values in `input` at the indices specified by `indices` using the values from `updates`, and saves the result to the output tensor. The data in `input` is modified.

## Prototype

```python
torch_npu.npu_scatter_nd_update_(input, indices, updates) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `updates`. The shape must have 1 to 8 dimensions.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - <term>Atlas inference products</term>: The data type can be `float32`, `float16`, or `bool`.
    - <term>Atlas training products</term>: The data type can be `float32`, `float16`, or `bool`.
      
- **`indices`** (`Tensor`): Required. Index tensor. The data type can be `int32` or `int64`. The data layout can be ND. Non-contiguous tensors are supported. The index values in `indices` must not be out of bounds.
- **`updates`** (`Tensor`): Required. Update data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `input`.
    - <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term>: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - <term>Atlas inference products</term>: The data type can be `float32`, `float16`, or `bool`.
    - <term>Atlas training products</term>: The data type can be `float32`, `float16`, or `bool`.
   
## Return Value

`Tensor`

Output tensor representing the results after `input` is updated.

## Constraints

- This API supports graph mode.
- `indices` must have at least 2 dimensions. The size of its last dimension must not exceed the rank of `input`.
- Assume that the size of the last dimension of `indices` is `a`. The shape of `updates` must be identical to that of `indices` excluding the last dimension combined with the shape of `input` excluding the first `a` dimensions. For example, if the shape of `input` is `(4, 5, 6)` and that of `indices` is `(3, 2)`, the shape of `updates` must be `(3, 6)`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import numpy as np
    >>>
    >>> data_var = np.random.uniform(0, 1, [24, 128]).astype(np.float16)
    >>> var = torch.from_numpy(data_var).to(torch.float16).npu()
    >>>    
    >>> data_indices = np.random.uniform(0, 12, [12, 1]).astype(np.int32)
    >>> indices = torch.from_numpy(data_indices).to(torch.int32).npu()
    >>>
    >>> data_updates = np.random.uniform(1, 2, [12, 128]).astype(np.float16)
    >>> updates = torch.from_numpy(data_updates).to(torch.float16).npu()
    >>>
    >>> out=torch_npu.npu_scatter_nd_update_(var, indices, updates)
    >>> print(out)
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
            # Calling the target API
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
    
    # Pass parameters
    out=npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu())
    print(out.shape, out.dtype)

    # Expected output of the preceding code sample:
    torch.Size([33, 5]) torch.float16
    ```
