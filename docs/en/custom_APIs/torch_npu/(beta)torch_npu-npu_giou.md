# (beta) torch_npu.npu_giou

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

First computes the minimum enclosing area and IoU of two boxes, then calculates the proportion of the enclosing area that does not belong to either box, and finally subtracts this proportion from the IoU to obtain the GIoU.

## Prototype

```python
torch_npu.npu_giou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
```

## Parameters

- `self` (`Tensor`): Required. Labeled bounding boxes. This parameter must be a 2D tensor with shape `(N, 4)`. The data type can be `float16` or `float32`. `N` indicates the number of labeled bounding boxes, and `4` indicates `[x1, y1, x2, y2]` or `[x, y, w, h]`.
- **`gtboxes`** (`Tensor`): Required. Ground-truth bounding boxes. This parameter must be a 2D tensor with shape `(M, 4)`. The data type can be `float16` or `float32`. `M` indicates the number of ground-truth bounding boxes, and `4` indicates `[x1, y1, x2, y2]` or `[x, y, w, h]`.
- **`trans`** (`bool`): Optional. A value of `True` indicates `"xywh"` and a value of `False` indicates `"xyxy"`. The default value is `False`.
- **`is_cross`** (`bool`): Optional. Controls whether the output shape is `(M, N)` or `(1, N)`. If set to `True`, the output shape must be `(M, N)`. If set to `False`, the output shape must be `(1, N)`. The default value is `False`.
- **`mode`** (`int`): Optional. Computation mode. Valid values are `0` (IoU) or `1` (IoF). The default value is `0`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> import numpy as np
>>> a=np.random.uniform(0,1,(10,4)).astype(np.float16)
>>> b=np.random.uniform(0,1,(10,4)).astype(np.float16)
>>> box1=torch.from_numpy(a).to("npu")
>>> box2=torch.from_numpy(b).to("npu")
>>> output = torch_npu.npu_giou(box1, box2, trans=True, is_cross=False, mode=0)
```
