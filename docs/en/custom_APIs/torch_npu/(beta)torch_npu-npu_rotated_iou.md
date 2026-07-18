# (beta) torch_npu.npu_rotated_iou

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

Computes the IoU of rotated bounding boxes.

## Prototype

```python
torch_npu.npu_rotated_iou(self, query_boxes, trans=False, mode=0, is_cross=True,v_threshold=0.0, e_threshold=0.0) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Gradient increment data. This parameter must be a 3D `float32` tensor with shape `(B, 5, N)`.
- **`query_boxes`** (`Tensor`): Required. Labeled bounding boxes. This parameter must be a 3D `float32` tensor with shape `(B, 5, K)`. `K` must not exceed 1600.
- **`trans`** (`bool`): Optional. A value of `True` indicates `"xyxyt"` and a value of `False` indicates `"xywht"`. The default value is `False`.
- **`is_cross`** (`bool`): Optional. `True` applies cross computation and `False` applies one-to-one computation. The default value is `True`.
- **`mode`** (`int`): Optional. Computation mode. Valid values are `0` (IoU) or `1` (IoF). The default value is `0`.
- **`v_threshold`** (`float`): Optional. Height threshold for rotated bounding boxes. The default value is `0.0`.
- **`e_threshold`** (`float`): Optional. Angle threshold for rotated bounding boxes. The default value is `0.0`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(2,2,5)).astype(np.float32)
>>> b=np.random.uniform(0,1,(2,3,5)).astype(np.float32)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(b).to("npu")
>>> output = torch_npu.npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
```
