# (beta) torch_npu.contrib.npu_giou

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

Provides an NPU-based Generalized Intersection over Union (GIoU) computation API.

## Prototype

```python
torch_npu.contrib.npu_giou(boxes1, boxes2, is_permuted=True)
```

## Parameters

- **`boxes1`** (`Tensor`): Predicted bounding boxes. This parameter must be 2D with shape `(n, 4)` in the `xywh` format.
- **`boxes2`** (`Tensor`): Corresponding ground truth bounding boxes. This parameter must be 2D with shape `(n, 4)`.
- **`is_permuted`** (`bool`): Indicates whether the coordinate values have been permuted. The default value is `True`.

## Example

```python
>>> import torch, torch_npu
>>> box1 = torch.randn(32, 4).npu()
>>> box1.requires_grad = True      
>>> box2 = torch.randn(32, 4).npu()
>>> box2.requires_grad = True
>>> iou = torch_npu.contrib.npu_giou(box1, box2)
>>> l = iou.sum()
>>> l.backward()
```
