# (beta) torch_npu.npu_ciou

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

NPU-based CIoU operation, which computes the CIoU (Complete Intersection over Union) loss between the predicted bounding boxes and the ground truth bounding boxes.

## Prototype

```python
torch_npu.npu_ciou(boxes1, boxes2, trans=False, is_cross=True, mode=0, atan_sub_flag=False) -> Tensor
```

## Parameters

- **`boxes1`** (`Tensor`): Required. Predicted bounding boxes in `xywh` format. This parameter must be a 2D tensor with shape `(4, n)`.
- **`boxes2`** (`Tensor`): Required. Ground-truth bounding boxes. This parameter must be a 2D tensor with shape `(4, n)`.
- **`trans`** (`bool`): Optional. Indicates whether there are offsets. The default value is `False`.
- **`is_cross`** (`bool`): Optional. Indicates whether a cross operation is performed between `boxes1` and `boxes2`. The default value is `True`.
- **`mode`** (`int`): Optional. CIoU computation mode. Valid values are `0` (IoU) or `1` (IoF). The default value is `0`.
- **`atan_sub_flag`** (`bool`): Optional. Specifies whether to pass the second value of the forward computation to the backward computation. The default value is `False`.

## Return Values

`Tensor`

CIoU computation result.

## Constraints

<!-- npu="950" id6 -->
- Ascend 950DT products: The second dimension of `boxes1` or `boxes2` can only be a multiple of 1024, `is_cross` can only be False, and the backward computation is not supported yet.
<!-- end id6 -->
<!-- npu="A3,910b,910,310p" id7 -->
- Atlas A3 training products, Atlas A2 training products, Atlas inference products, and Atlas training products: So far, the CIoU backward computation supports only trans==True, is_cross==False, and mode==0 (iou). If backward propagation is required, ensure that the parameters are correct.
<!-- end id7 -->

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
