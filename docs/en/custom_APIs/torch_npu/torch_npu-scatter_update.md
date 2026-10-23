# torch_npu.scatter_update

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
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Updates the values in tensor `data` with the values in tensor `updates` according to the specified `axis` and `indices`, and saves the result to the output tensor. The data in `data` itself is not changed.

## Prototype

```python
torch_npu.scatter_update(data, indices, updates, axis) -> Tensor
```

## Parameters

- **`data`** (`Tensor`): Required. Original data before the update. This parameter must be 2D to 8D, and the dimensions must match those of `updates`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.

    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id5 -->
    <!-- npu="A3" id6 -->
    - <term>Atlas A3 training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id6 -->
    <!-- npu="910" id7 -->
    - <term>Atlas training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, or `torch.int32`.
    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.uint8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id8 -->

- **`indices`** (`Tensor`): Required. Indices used for the update. The data type can be `torch.int32` or `torch.int64`. This parameter must be 1D or 2D. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`updates`** (`Tensor`): Required. Data used for the update. The dimensions of `updates` must match those of `data`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id10 -->
    <!-- npu="910" id11 -->
    - <term>Atlas training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, or `torch.int32`.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.uint8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
    <!-- end id12 -->

- **`axis`** (`int`): Required. Axis that specifies the dimension on which the scatter update is performed. The data type is `int64`. The value range is (-`data_rank`, `data_rank`) (where `data_rank` is the number of dimensions of `data`), and `axis` cannot be 0.

## Return Values

`Tensor`

Computes the output. Only 2D to 8D tensors are supported, and the dimension sizes must be the same as those of `data`. Non-contiguous tensors are supported. The data layout can be $ND$. Empty tensors are not supported.

<!-- npu="910b" id13 -->
- <term>Atlas A2 products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
<!-- end id13 -->
<!-- npu="A3" id14 -->
- <term>Atlas A3 training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
<!-- end id14 -->
<!-- npu="910" id15 -->
- <term>Atlas training products</term>: The data type can be `torch.int8`, `torch.float16`, `torch.float32`, or `torch.int32`.
<!-- end id15 -->
<!-- npu="950" id16 -->
- <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.uint8`, `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int32`.
<!-- end id16 -->

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- `data` and `updates` must have the same rank.
- Out-of-range indices are not supported. The user must ensure that the indices are valid; the framework does not perform bounds checking.
- When the operation is update and `indices` contains duplicates, the result at the duplicate positions is not guaranteed.
- The 0th axis of the `updates` shape must be the same as the 0th axis of the `indices` shape.
- When `indices` is 0D, the 0th axis of the `updates` shape is 1.
- The 0th axis of the `updates` shape must be less than or equal to the 0th axis of the `data` shape.
- Except for the `axis` axis and the 0th axis, the shapes of `updates` and `data` must be the same on all other axes.
- When the `indices` shape is 2D, the 1st axis of the shape must be 2.

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype = torch.float32).npu()
    indices = torch.tensor ([1],dtype = torch.int64).npu()
    updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]], dtype = torch.float32).npu()
    out = torch_npu.scatter_update(data, indices, updates, axis = -2)
    ```

<!-- npu="950" id17 -->
- Graph mode call: Only <term>Ascend 950PR&950DT products</term> are supported.

    ```python
    import os
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch.nn as nn
    import torch
    import numpy as np
    import numpy
    torch_npu.npu.set_compile_mode(jit_compile = True)

    os.environ["ENABLE_ACLNN"] = "false"
    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, var, indices, update, axis):
            res = torch_npu.scatter_update(var, indices, update, axis)
            return res

    npu_mode = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config = config)
    npu_mode = torch.compile(npu_mode, fullgraph = True, backend = npu_backend, dynamic=False)

    dtype = np.float32
    x = [37,58]
    indices = [31]
    update = [31,22]
    axis = 1

    data_x = np.random.uniform(0, 1, x).astype(dtype)
    data_indices = np.random.uniform(0, 10, indices).astype(dtype)
    data_update = np.random.uniform(0, 1, update).astype(dtype)

    tensor_x = torch.from_numpy(data_x).to(torch.float32)
    tensor_indices = torch.from_numpy(data_indices).to(torch.int32)
    tensor_update = torch.from_numpy(data_update).to(torch.float32)

    # Pass the parameters
    print(npu_mode(tensor_x.npu(), tensor_indices.npu(), tensor_update.npu(), axis))
    ```
<!-- end id17 -->
