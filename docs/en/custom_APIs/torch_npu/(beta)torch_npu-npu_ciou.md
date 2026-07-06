# (beta) torch_npu.npu_ciou

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

Applies an NPU-based CIoU operation. CIoU is formulated by introducing a penalty term to DIoU.

## Prototype

```python
torch_npu.npu_ciou(self, gtboxes, trans=False, is_cross=True, mode=0, atan_sub_flag=False) -> Tensor
```

## Parameters

- **`boxes1`** (`Tensor`): Required. Predicted bounding boxes in `xywh` format. This parameter must be a 2D tensor with shape `(4, n)`.
- **`boxes2`** (`Tensor`): Required. Ground-truth bounding boxes. This parameter must be a 2D tensor with shape `(4, n)`.
- **`trans`** (`bool`): Optional. Indicates whether there are offsets. The default value is `False`.
- **`is_cross`** (`bool`): Optional. Indicates whether a cross operation is performed between `box1` and `box2`. The default value is `True`.
- **`mode`** (`int`): Optional. CIoU computation mode. Valid values are `0` (IoU) or `1` (IoF). The default value is `0`.
- **`atan_sub_flag`** (`bool`): Optional. Specifies whether to pass the second value of the forward computation to the backward computation. The default value is `False`.

## Return Values

`Tensor`

Result of the mask operation.

## Constraints

Currently, CIoU backward computation only supports `trans == True`, `is_cross == False`, and `mode == 0` ('iou'). If backward propagation is required, ensure that the parameters are correct.

## Example

```python
>>> box1 = torch.randn(4, 32).npu()
>>> box1.requires_grad = True
>>> box2 = torch.randn(4, 32).npu()
>>> box2.requires_grad = True
>>> ciou = torch_npu.npu_ciou(box1, box2, trans=True, is_cross=False, mode=0)
>>> l = ciou.sum()
>>> l.backward()
```
