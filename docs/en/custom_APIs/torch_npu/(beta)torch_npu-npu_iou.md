# (beta) torch_npu.npu_iou

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

Computes the intersection over union (IoU) or intersection over foreground (IoF) based on the ground truth and predicted regions.

## Prototype

```python
torch_npu.npu_iou(bboxes, gtboxes, mode=0) -> Tensor 
```

## Parameters

- **`bboxes`** (`Tensor`): Required. Input tensor.
- **`gtboxes`** (`Tensor`): Required. Input tensor.
- **`mode`** (`int`): Optional. Valid values are `0` (IoU mode) or `1` (IoF mode). The default value is `0`.

<!-- npu="950" id6 -->
## Constraints

Ascend 950DT products: Backward computation is not supported for this API currently.
<!-- end id6 -->

## Example

```python
>>> import torch
>>> import torch_npu
>>> bboxes = torch.tensor([[0, 0, 10, 10],
                           [10, 10, 20, 20],
                           [32, 32, 38, 42]], dtype=torch.float16).to("npu")
>>> gtboxes = torch.tensor([[0, 0, 10, 20],
                            [0, 10, 10, 10],
                            [10, 10, 20, 20]], dtype=torch.float16).to("npu")
>>> output_iou = torch_npu.npu_iou(bboxes, gtboxes, 0)
>>> print(output_iou)
tensor([[0.4985, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000], 
       [0.0000, 0.9961, 0.0000]], device='npu:0', dtype=torch.float16)
```
