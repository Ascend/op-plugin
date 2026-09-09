# (beta) torch_npu.npu_rotated_overlaps

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Computes the overlap area of rotated bounding boxes.

## Prototype

```python
torch_npu.npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Gradient increment data. This parameter must be a 3D `float32` tensor with shape `(B, 5, N)`.
- **`query_boxes`** (`Tensor`): Required. Labeled bounding boxes. This parameter must be a 3D `float32` tensor with shape `(B, 5, K)`.
- **`trans`** (`bool`): Optional. A value of `True` indicates `"xyxyt"` and a value of `False` indicates `"xywht"`. The default value is `False`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(1,3,5)).astype(np.float32)
>>> b=np.random.uniform(0,1,(1,2,5)).astype(np.float32)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(b).to("npu")
>>> output = torch_npu.npu_rotated_overlaps(box1, box2, trans=False)
>>> print(output)
tensor([[[0.0000, 0.1562, 0.0000],
        [0.1562, 0.3713, 0.0611],
        [0.0000, 0.0611, 0.0000]]], device='npu:0', dtype=torch.float32)
```
