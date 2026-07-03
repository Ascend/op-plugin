# (beta) torch_npu.contrib.function.npu_ciou

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

Applies an NPU-based CIoU operation. CIoU is formulated by introducing a penalty term to DIoU.

## Prototype

```python
torch_npu.contrib.function.npu_ciou(boxes1, boxes2, trans=True, is_cross=False, mode=0)
```

## Parameters

- **`boxes1`** (`Tensor`): Predicted bounding boxes in `xywh` format. This parameter must be 2D with shape `(4, n)`.
- **`boxes2`** (`Tensor`): Corresponding ground truth bounding boxes. This parameter must be 2D with shape `(4, n)`.
- **`trans`** (`bool`): Indicates whether there are offsets. The default value is `False`.
- **`is_cross`** (`bool`): Indicates whether a cross operation is performed between `box1` and `box2`. The default value is `False`.
- **`mode`** (`int`): CIoU computation mode. The default value is `0`.

## Return Values

`Tensor` 

 IoU values with shape `[1,n]`.

## Constraints

Currently, CIoU backward computation only supports `trans == True`, `is_cross == False`, and `mode == 0` (`'iou'`). If backward propagation is required, ensure that the parameters are correct.

## Example

```python
>>> from torch_npu.contrib.function import npu_ciou
>>> box1 = torch.randn(4, 32).npu()
>>> box1.requires_grad = True
>>> box2 = torch.randn(4, 32).npu()
>>> box2.requires_grad = True
>>> ciou = npu_ciou(box1, box2) 
>>> l = ciou.sum()
>>> l.backward()
```
