# (beta) torch_npu.contrib.npu_ptiou

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

Provides the NPU version of PTIoU computation operations. During computation, no minimum value is added to the overlapping area.

## Prototype

```python
torch_npu.contrib.npu_ptiou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## Parameters

- **`boxes1`** (`Tensor`): Predicted bounding boxes. This parameter must be 2D with shape `(n, 4)`.
- **`boxes2`** (`Tensor`): Predicted bounding boxes. This parameter must be 2D with shape `(m, 4)`.
- **`is_normalized`** (`bool`): Indicates whether the coordinate values have been normalized. The default value is `False`.
- **`normalized_scale`** (`float`): Normalization scale for restoring coordinates. The default value is `100`.

## Constraints

This function is commonly used for matching bounding boxes and anchors. There is currently no corresponding backward operator, so this function cannot be used for `IOU_Loss`. The computation formula adds `0.001` to the denominator to avoid division by zero. When the input bounding boxes are normalized, the `0.001` term may have an excessive impact. You are advised to scale up the input values to reduce the impact of `0.001`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> box1 = torch.randint(0, 256, size=(32, 4)).npu()
>>> box2 = torch.randint(0, 256, size=(16, 4)).npu()
>>> iou = torch_npu.contrib.npu_ptiou(box1, box2)
```
