# torch_npu.npu_scatter_nd_update

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Updates the values in `input` at the specified `indices` using the values from `updates`, and saves the result to the output tensor. The data in `input` remains unchanged.

## Prototype

```python
torch_npu.npu_scatter_nd_update(input, indices, updates) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Source data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `updates`. The shape must have 1 to 8 dimensions.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - Atlas inference products: The data type can be `float32`, `float16`, or `bool`.
    - Atlas training products: The data type can be `float32`, `float16`, or `bool`.
      
- **`indices`** (`Tensor`): Required. Index tensor. The data type can be `int32` or `int64`. The data layout can be ND. Non-contiguous tensors are supported. The index values in `indices` must not be out of bounds.
- **`updates`** (`Tensor`): Required. Update data tensor. The data layout can be ND. Non-contiguous tensors are supported. The data type must be identical to that of `input`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32`, `float16`, `bool`, `bfloat16`, `int64`, or `int8`.
    - Atlas inference products: The data type can be `float32`, `float16`, or `bool`.
    - Atlas training products: The data type can be `float32`, `float16`, or `bool`.

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
    >>> out = torch_npu.npu_scatter_nd_update(var, indices, updates)
    >>> print(out)
    tensor([[0.6475, 0.3469, 0.2915,  ..., 0.7368, 0.8301, 0.1155],
            [0.5308, 0.7754, 0.5967,  ..., 0.2219, 0.0421, 0.2339],
            [1.7646, 1.1406, 1.5127,  ..., 1.3438, 1.8018, 1.0361],
            ...,
            [0.6396, 0.5396, 0.2939,  ..., 0.9409, 0.5161, 0.1169],
            [0.0737, 0.0457, 0.4727,  ..., 0.5068, 0.8418, 0.6104],
            [0.4180, 0.9102, 0.1122,  ..., 0.0540, 0.4041, 0.3889]],
        device='npu:0', dtype=torch.float16)
    >>> print(out.shape)
    torch.Size([24, 128])
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
            res = torch_npu.npu_scatter_nd_update(var, indices, update)
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
    print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu()))


    # Expected output of the preceding code sample:  
    tensor([[0.6602, 0.4719, 0.8823, 0.8369, 0.8833],
        [0.7993, 0.2986, 0.0251, 0.8555, 0.7559],
        [0.1278, 0.9434, 0.9409, 0.0586, 0.1710],
        ...,
        [0.9399, 0.8940, 0.5708, 0.7319, 0.1566],
        [0.1333, 0.9614, 0.6128, 0.8457, 0.0269],
        [0.2491, 0.0362, 0.5776, 0.6094, 0.1281],
        [0.2092, 0.7417, 0.8862, 0.1210, 0.8130],
        [0.2910, 0.2468, 0.5488, 0.9761, 0.9785]], device='npu:0',
       dtype=torch.float16)
    ```
