# (beta) torch_npu.npu_diou

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

Applies an NPU-based DIoU operation. Considering the distance between targets and the overlap ratio of distance and scope, different targets or boundaries must tend to be stable.

## Prototype

```python
torch_npu.npu_diou(self, gtboxes, trans=False, is_cross=False, mode=0) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Predicted bounding boxes in `xywh` format. This parameter must be a 2D tensor with shape `(4, n)`.
- **`gtboxes`** (`Tensor`): Ground-truth bounding boxes. This parameter must be a 2D tensor with shape `(4, n)`.
- **`trans`** (`bool`): Optional. Indicates whether there are offsets. The default value is `False`.
- **`is_cross`** (`bool`): Optional. Specifies whether to perform a cross operation between `box1` and `box2`. The default value is `False`.
- **`mode`** (`int`): DIoU computation mode. Valid values are `0` (IoU) or `1` (IoF). The default value is `0`.

## Return Values

`Tensor`

Result of the mask operation.

## Constraints

Currently, DIoU backward computation only supports `trans == True`, `is_cross == False`, and `mode == 0` ('iou') in the current version. If backward propagation is required, ensure that the parameters are correct.

## Example

```python
    >>> box1 = torch.randn(4, 32).npu()
    >>> box1.requires_grad = True
    >>> box2 = torch.randn(4, 32).npu()
    >>> box2.requires_grad = True
    >>> diou = torch_npu.contrib.function.npu_diou(box1, box2) 
    >>> l = diou.sum()
    >>> l.backward()
```
