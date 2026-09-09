# (beta) torch_npu.contrib.npu_iou

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

Provides the NPU version of IoU computation operations. During computation, a small value is added to the overlapping area to avoid division-by-zero errors.

## Prototype

```python
torch_npu.contrib.npu_iou(boxes1, boxes2, mode="ptiou", is_normalized=False, normalized_scale=100.)
```

## Parameters

- **`boxes1`** (`Tensor`): Predicted bounding boxes. This parameter must be 2D with shape `(n, 4)`.
- **`boxes2`** (`Tensor`): Predicted bounding boxes. This parameter must be 2D with shape `(m, 4)`.
- **`is_normalized`** (`bool`): Indicates whether the coordinate values have been normalized. Default value: `False`.
- **`normalized_scale`** (`float`): Normalization scale for restoring coordinates. The default value is `100`.

## Constraints

This function is commonly used for matching bounding boxes and anchors. This function lacks a corresponding backward computation operator. Therefore, it cannot be used in `IoU_Loss`. Because the computation formula adds 0.001 to the denominator to avoid division by zero, the 0.001 component becomes too large when input bounding boxes contain normalized data. You must scale up input values to prevent 0.001 from causing excessive impact.

## Example

```python
>>> box1 = torch.randint(0, 256, size=(32, 4)).npu()
>>> box2 = torch.randint(0, 256, size=(16, 4)).npu()
>>> iou = torch_npu.contrib.npu_iou(box1, box2)
```
