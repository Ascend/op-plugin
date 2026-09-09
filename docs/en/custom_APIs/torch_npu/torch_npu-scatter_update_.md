# torch_npu.scatter_update_

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |
|<term>Atlas training products</term>   | √  |

## Function

Updates values in the `data` tensor with values from the `updates` tensor according to the specified `axis` and `indices`, and saves the results to an output tensor. The data within the original `data` tensor is modified in-place.

## Prototype

```python
torch_npu.scatter_update_(data, indices, updates, axis) -> Tensor
```

## Parameters

- **`data`** (`Tensor`): Required. Original data before the update. This parameter must be 2D to 8D, and the dimensions must match those of `updates`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
    - Atlas A3 training products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
    - Atlas training products: The data type can be `int8`, `float16`, `float32`, or `int32`.

- **`indices`** (`Tensor`): Required. Indices used for the update. The data type can be `int32` or `int64`. This parameter must be 1D or 2D. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`updates`** (`Tensor`): Required. Data used for the update. The dimensions of `updates` must match those of `data`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
    - Atlas A3 training products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
    - Atlas training products: The data type can be `int8`, `float16`, `float32`, or `int32`.

- **`axis`** (`int`): Required. Axis that specifies the dimension on which the scatter update is performed. The data type can be `int64`.

## Return Values

`Tensor`

Computation result, which reuses the input memory address. This parameter must be 2D to 8D, and the dimensions must match those of `data`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are not supported.

- Atlas A2 training products/Atlas A2 inference products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
- Atlas A3 training products: The data type can be `int8`, `float16`, `float32`, `bfloat16`, or `int32`.
- Atlas training products: The data type can be `int8`, `float16`, `float32`, or `int32`.

## Constraints

- `data` and `updates` must have the same rank.
- Out-of-bounds indices are not supported. Ensure that all indices are valid, as the framework does not perform bounds checking.

## Example

```python
import torch
import torch_npu
import numpy as np
data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype=torch.float32).npu()
indices = torch.tensor ([1],dtype=torch.int64).npu()
updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]] , dtype=torch.float32).npu()
out = torch_npu.scatter_update_(data, indices, updates, axis=-2)
```
